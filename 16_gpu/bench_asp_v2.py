"""
TEST DÉCISIF ASP — version 2, avec les DEUX schémas implémentés de façon optimale.

Ce que la v1 a révélé. Le gather de la passe 2 atteint 52-85 GB/s, c'est-à-dire la
vitesse du contigu : il ne coûte quasiment rien. Le goulot était la passe 1, qui est
pourtant un accès CONTIGU — mais que la v1 implémentait avec un work-group par bloc,
soit des morceaux de 128 a 512 octets, régime où le GPU ne sature pas (13-53 GB/s).
C'était un artefact du banc, pas une propriété d'ASP.

Correction. Les lectures contiguës (le schéma plat, et la passe 1 d'ASP) sont
désormais faites par un noyau de FLUX : chaque work-group balaie un grand segment
contigu (16 Kio), indépendamment du découpage logique en blocs. Seule la passe 2
d'ASP conserve un work-group par bloc, car son irrégularité est irréductible.

C'est la comparaison loyale : chaque schéma est implémenté au mieux de ce que son
motif d'accès permet.
"""
import numpy as np, pyopencl as cl, json, pathlib, argparse

OUT = pathlib.Path(__file__).resolve().parent / "resultats"
OUT.mkdir(exist_ok=True)

SRC = """
// Lecture CONTIGUE en flux : chaque groupe balaie `span` uint4 consecutifs.
__kernel void stream(__global const uint4* src, const uint span,
                     __global uint4* sink)
{
    const uint g = get_group_id(0), lid = get_local_id(0), ls = get_local_size(0);
    const uint base = g * span;
    uint4 acc = (uint4)(0,0,0,0);
    for (uint i = lid; i < span; i += ls) acc ^= src[base + i];
    __local uint4 red[256];
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = ls >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] ^= red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) sink[g] = red[0];
}

// Lecture DISPERSEE : un groupe par morceau, offsets arbitraires.
__kernel void gather(__global const uint4* src, __global const uint* off,
                     const uint chunk, __global uint4* sink)
{
    const uint g = get_group_id(0), lid = get_local_id(0), ls = get_local_size(0);
    const uint base = off[g];
    uint4 acc = (uint4)(0,0,0,0);
    for (uint i = lid; i < chunk; i += ls) acc ^= src[base + i];
    __local uint4 red[256];
    red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = ls >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] ^= red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) sink[g] = red[0];
}
"""
LS = 64
SPAN_BYTES = 16384          # segment contigu par work-group, pour le noyau de flux


def t_stream(q, k, src, nbytes, sink, reps):
    span_u4 = SPAN_BYTES // 16
    ng = max(nbytes // SPAN_BYTES, 1)
    ts = []
    for _ in range(reps):
        k.set_args(src, np.uint32(span_u4), sink)
        ev = cl.enqueue_nd_range_kernel(q, k, (ng * LS,), (LS,))
        ev.wait(); ts.append((ev.profile.end - ev.profile.start) * 1e-9)
    return float(np.median(ts[2:]))


def t_gather(q, k, src, offb, chunk_u4, ng, sink, reps):
    ts = []
    for _ in range(reps):
        k.set_args(src, offb, np.uint32(chunk_u4), sink)
        ev = cl.enqueue_nd_range_kernel(q, k, (ng * LS,), (LS,))
        ev.wait(); ts.append((ev.profile.end - ev.profile.start) * 1e-9)
    return float(np.median(ts[2:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--reps", type=int, default=13)
    ap.add_argument("--passes", type=int, default=2, help="repetitions du balayage complet")
    a = ap.parse_args()

    dev = next(d for p in cl.get_platforms() for d in p.get_devices()
               if d.type & cl.device_type.GPU)
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prg = cl.Program(ctx, SRC).build(options="-cl-std=CL2.0")
    kst, kga = cl.Kernel(prg, "stream"), cl.Kernel(prg, "gather")
    mf = cl.mem_flags

    print("=" * 108)
    print(f"TEST DÉCISIF ASP v2  --  {dev.name}, {dev.max_compute_units} CU, "
          f"{dev.max_clock_frequency} MHz, pilote {dev.driver_version}")
    print("  contigu = noyau de flux (segments de 16 Kio) ; gather = un groupe par bloc")
    print("=" * 108)

    rng = np.random.default_rng(0)
    src_u4 = (1024 << 20) // 16
    host = rng.integers(0, 2**32, size=(src_u4, 4), dtype=np.uint32)
    srcb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host)
    del host
    sink = cl.Buffer(ctx, mf.WRITE_ONLY, size=(1 << 23))

    allrows = []
    for sweep in range(a.passes):
        rows = []
        if sweep == 0:
            print(f"\n{'n':>6s} {'D':>4s} {'r1':>3s} {'r2':>3s} {'m':>6s} | "
                  f"{'gain theo.':>10s} | {'t plat ms':>10s} {'t p1':>7s} {'t p2':>7s} "
                  f"{'t ASP':>7s} | {'GAIN REEL':>10s} {'efficacite':>11s}")
            print("-" * 108)
        for D in (a.D, 128):
            for n in (4096, 16384):
                for r1, r2 in ((2, 8), (2, 16), (4, 16), (4, 32)):
                    S1, S2 = r1 * D * 2, r2 * D * 2
                    c2 = S2 // 16
                    if c2 < 1 or n * c2 > src_u4:
                        continue
                    for frac in (4, 8):
                        m = n // frac
                        if m < 32:
                            continue
                        by_flat, by_p1, by_p2 = n * S2, n * S1, m * S2
                        t_flat = t_stream(q, kst, srcb, by_flat, sink, a.reps)
                        t_p1 = t_stream(q, kst, srcb, by_p1, sink, a.reps)
                        pick = rng.choice(n, size=m, replace=False).astype(np.uint32)
                        offb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                         hostbuf=pick * c2)
                        t_p2 = t_gather(q, kga, srcb, offb, c2, m, sink, a.reps)
                        offb.release()
                        t_asp = t_p1 + t_p2
                        g_theo = by_flat / (by_p1 + by_p2)
                        g_real = t_flat / t_asp
                        rows.append(dict(sweep=sweep, n=n, D=D, r1=r1, r2=r2, m=m,
                                         gain_theo=g_theo, t_flat=t_flat, t_p1=t_p1,
                                         t_p2=t_p2, gain_real=g_real,
                                         eff=g_real / g_theo,
                                         bw_gather=by_p2 / t_p2 / 1e9,
                                         bw_flat=by_flat / t_flat / 1e9))
                        if sweep == 0:
                            print(f"{n:6d} {D:4d} {r1:3d} {r2:3d} {m:6d} | "
                                  f"{g_theo:9.2f}x | {1e3*t_flat:10.3f} "
                                  f"{1e3*t_p1:7.3f} {1e3*t_p2:7.3f} {1e3*t_asp:7.3f} | "
                                  f"{g_real:9.2f}x {g_real/g_theo:10.1%}")
        allrows += rows

    e0 = np.array([r["eff"] for r in allrows if r["sweep"] == 0])
    e1 = np.array([r["eff"] for r in allrows if r["sweep"] == 1]) if a.passes > 1 else e0
    print("-" * 108)
    print(f"\n  EFFICACITÉ = gain réel / gain théorique en octets")
    print(f"    balayage 1 : médiane {np.median(e0):.1%}  [{e0.min():.1%} - {e0.max():.1%}]")
    if a.passes > 1:
        print(f"    balayage 2 : médiane {np.median(e1):.1%}  [{e1.min():.1%} - {e1.max():.1%}]"
              f"   (contrôle de dérive thermique : écart médian "
              f"{abs(np.median(e0)-np.median(e1)):.1%})")
    bad = [r for r in allrows if r["gain_real"] < 1]
    print(f"    configurations plus lentes que le plat : {len(bad)}/{len(allrows)}")
    bwg = np.array([r["bw_gather"] for r in allrows])
    bwf = np.array([r["bw_flat"] for r in allrows])
    print(f"\n  débit du GATHER  : médiane {np.median(bwg):.1f} GB/s")
    print(f"  débit du CONTIGU : médiane {np.median(bwf):.1f} GB/s")
    print(f"  -> le gather atteint {np.median(bwg)/np.median(bwf):.1%} du débit contigu")
    (OUT / "bench_asp_v2.json").write_text(json.dumps(
        dict(device=dev.name, cu=dev.max_compute_units, driver=dev.driver_version,
             rows=allrows), indent=2))
    print(f"\n-> {OUT/'bench_asp_v2.json'}")


if __name__ == "__main__":
    main()
