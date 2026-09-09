"""
TEST DÉCISIF — le gain d'ASP survit-il au gather irrégulier, mesuré sur GPU réel ?

On reproduit les DEUX motifs d'accès complets, à paramètres réalistes :

  PLAT  : lire le résumé fin de TOUS les n blocs, contigu
          -> n * S2 octets, un seul noyau contigu

  ASP   : passe 1, lire le résumé grossier de TOUS les n blocs, contigu
          passe 2, lire le résumé fin des m survivants, DISPERSÉ
          -> n * S1 + m * S2 octets, deux noyaux

On mesure le temps mural GPU des deux, et on compare le gain RÉEL au gain THÉORIQUE
en octets. Si gain_reel / gain_theorique ~ 1, le gather ne coûte rien ; s'il est
nettement < 1, il mange le bénéfice.

Paramètres réalistes. À 1M de contexte et blocs de 64 tokens, n = 15625 blocs.
Un résumé de rang r en fp16 pèse S = r*D*2 octets (D = dimension de tête).
"""
import numpy as np, pyopencl as cl, json, pathlib, argparse

OUT = pathlib.Path(__file__).resolve().parent / "resultats"
OUT.mkdir(exist_ok=True)

SRC = """
__kernel void read_chunks(__global const uint4* src,
                          __global const uint*  off,
                          const uint chunk,
                          __global uint4* sink)
{
    const uint g   = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint ls  = get_local_size(0);
    const uint base = off[g];
    uint4 acc = (uint4)(0,0,0,0);
    for (uint i = lid; i < chunk; i += ls) acc ^= src[base + i];
    __local uint4 red[256];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = ls >> 1; s > 0; s >>= 1) {
        if (lid < s) red[lid] ^= red[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) sink[g] = red[0];
}
"""


def time_pass(q, kern, src, offbuf, chunk_u4, ngroups, sink, lsize, reps):
    ts = []
    for _ in range(reps):
        kern.set_args(src, offbuf, np.uint32(chunk_u4), sink)
        ev = cl.enqueue_nd_range_kernel(q, kern, (ngroups * lsize,), (lsize,))
        ev.wait()
        ts.append((ev.profile.end - ev.profile.start) * 1e-9)
    return float(np.median(ts[2:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--reps", type=int, default=11)
    a = ap.parse_args()

    dev = next(d for p in cl.get_platforms() for d in p.get_devices()
               if d.type & cl.device_type.GPU)
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prg = cl.Program(ctx, SRC).build(options="-cl-std=CL2.0")
    kern = cl.Kernel(prg, "read_chunks")
    mf = cl.mem_flags

    print("=" * 104)
    print(f"TEST DÉCISIF ASP  --  {dev.name}, {dev.max_compute_units} CU, "
          f"{dev.max_clock_frequency} MHz, pilote {dev.driver_version}")
    print("=" * 104)

    rng = np.random.default_rng(0)
    # buffer source : assez grand pour contenir n blocs de résumé fin r=32
    SRC_MB = 1024
    src_u4 = (SRC_MB << 20) // 16
    host = rng.integers(0, 2**32, size=(src_u4, 4), dtype=np.uint32)
    srcb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host)
    del host
    sink = cl.Buffer(ctx, mf.WRITE_ONLY, size=(1 << 22))

    rows = []
    print(f"\n{'n blocs':>8s} {'r1':>3s} {'r2':>3s} {'m':>7s} | "
          f"{'octets plat':>12s} {'octets ASP':>11s} {'gain theo.':>10s} | "
          f"{'t plat ms':>10s} {'t ASP ms':>9s} {'GAIN REEL':>10s} {'efficacite':>11s}")
    print("-" * 104)

    for n in (4096, 16384):
        for r1, r2 in ((1, 8), (2, 8), (2, 16), (4, 16)):
            S1, S2 = r1 * a.D * 2, r2 * a.D * 2         # octets par bloc
            c1, c2 = S1 // 16, S2 // 16                 # en uint4
            if c1 < 1 or c2 < 1:
                continue
            for frac in (4, 8):
                m = n // frac
                if n * c2 > src_u4 or m < 32:
                    continue
                # --- plat : n morceaux fins, contigus
                off_flat = (np.arange(n, dtype=np.uint32) * c2)
                bf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=off_flat)
                t_flat = time_pass(q, kern, srcb, bf, c2, n, sink, 64, a.reps)
                bf.release()
                # --- ASP passe 1 : n morceaux grossiers, contigus
                off_p1 = (np.arange(n, dtype=np.uint32) * c1)
                b1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=off_p1)
                t_p1 = time_pass(q, kern, srcb, b1, c1, n, sink, 64, a.reps)
                b1.release()
                # --- ASP passe 2 : m morceaux fins, DISPERSÉS parmi les n
                pick = rng.choice(n, size=m, replace=False).astype(np.uint32)
                off_p2 = pick * c2
                b2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=off_p2)
                t_p2 = time_pass(q, kern, srcb, b2, c2, m, sink, 64, a.reps)
                b2.release()

                by_flat = n * S2
                by_asp = n * S1 + m * S2
                g_theo = by_flat / by_asp
                t_asp = t_p1 + t_p2
                g_real = t_flat / t_asp
                eff = g_real / g_theo
                print(f"{n:8d} {r1:3d} {r2:3d} {m:7d} | {by_flat/1e6:9.2f} Mo "
                      f"{by_asp/1e6:8.2f} Mo {g_theo:9.2f}x | "
                      f"{1e3*t_flat:10.3f} {1e3*t_asp:9.3f} {g_real:9.2f}x "
                      f"{eff:10.1%}")
                rows.append(dict(n=n, r1=r1, r2=r2, m=m, D=a.D,
                                 bytes_flat=by_flat, bytes_asp=by_asp,
                                 gain_theo=g_theo, t_flat=t_flat, t_p1=t_p1,
                                 t_p2=t_p2, gain_real=g_real, eff=eff))
    effs = np.array([r["eff"] for r in rows])
    print("-" * 104)
    print(f"\n  EFFICACITÉ = gain réel / gain théorique en octets")
    print(f"  médiane {np.median(effs):.1%}   min {effs.min():.1%}   max {effs.max():.1%}")
    print(f"  configurations où ASP est plus LENT que le plat : "
          f"{sum(1 for r in rows if r['gain_real'] < 1)}/{len(rows)}")
    (OUT / "bench_asp.json").write_text(json.dumps(
        dict(device=dev.name, cu=dev.max_compute_units, driver=dev.driver_version,
             D=a.D, rows=rows), indent=2))
    print(f"\n-> {OUT/'bench_asp.json'}")


if __name__ == "__main__":
    main()
