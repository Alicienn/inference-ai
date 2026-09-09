"""
BANC DE MESURE — le gather irrégulier annule-t-il le gain en octets d'ASP ?

CONTEXTE. ASP (voir 12_poc/CONCEPT_ASP.md) lit en passe 1 un résumé grossier de TOUS
les n blocs (accès contigu), puis en passe 2 un résumé fin des seuls m survivants
(accès DISPERSÉ, car les survivants dépendent de la requête). Le gain théorique en
octets est de 1,1x a 2,4x. La question ouverte : le surcoût du gather irrégulier
l'annule-t-il ?

POINT CLÉ DE CONCEPTION, souvent mal posé. On parle ici d'un gather dont la GRANULARITÉ
n'est pas de 4 octets mais d'un RÉSUMÉ DE BLOC ENTIER : r vecteurs de dimension D en
fp16, soit r*D*2 octets. Pour r=8, D=64 -> 1024 octets, c'est-à-dire 16 lignes de cache
de 64 octets. La coalescence fonctionne donc PARFAITEMENT À L'INTÉRIEUR de chaque
morceau ; seule la localité entre morceaux est perdue. C'est ce que ce banc quantifie.

PROTOCOLE. Deux noyaux au flux d'instructions IDENTIQUE, ne différant que par le motif
d'offsets :
  - contigu : les G morceaux sont consécutifs
  - gather  : les G morceaux sont tirés au hasard, distincts, dans un buffer bien plus
              grand (facteur d'étalement `spread`)
Les deux lisent EXACTEMENT le même volume V. On balaie la taille de morceau C et le
facteur d'étalement. Buffers >> caches pour rester borné par la DRAM.
"""
import numpy as np, pyopencl as cl, json, pathlib, time, argparse

OUT = pathlib.Path(__file__).resolve().parent / "resultats"
OUT.mkdir(exist_ok=True)

SRC = """
// Un work-group traite un morceau. Les threads du groupe lisent le morceau de
// facon contigue -> coalescence parfaite intra-morceau dans les deux cas.
__kernel void read_chunks(__global const uint4* src,
                          __global const uint*  off,      // offset de chaque morceau, en uint4
                          const uint chunk,               // taille du morceau, en uint4
                          __global uint4* sink)
{
    const uint g   = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint ls  = get_local_size(0);
    const uint base = off[g];
    uint4 acc = (uint4)(0,0,0,0);
    for (uint i = lid; i < chunk; i += ls) {
        acc ^= src[base + i];
    }
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


def setup():
    plats = cl.get_platforms()
    dev = None
    for p in plats:
        for d in p.get_devices():
            if d.type & cl.device_type.GPU:
                dev = d; break
        if dev: break
    if dev is None:
        raise SystemExit("aucun device GPU OpenCL")
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prg = cl.Program(ctx, SRC).build(options="-cl-std=CL2.0")
    return dev, ctx, q, prg


def bench(ctx, q, prg, src_buf, src_u4, chunk_u4, n_chunks, spread, rng,
          reps=12, lsize=64):
    """Renvoie (temps_median_s, temps_min_s) pour lire n_chunks morceaux."""
    V_u4 = chunk_u4 * n_chunks
    # ---- offsets contigus
    off_c = (np.arange(n_chunks, dtype=np.uint32) * chunk_u4)
    # ---- offsets dispersés : morceaux distincts tirés dans une zone `spread` fois
    #      plus grande, puis melanges
    span = min(int(n_chunks * spread), src_u4 // chunk_u4)
    pick = rng.choice(span, size=n_chunks, replace=False).astype(np.uint32)
    off_g = pick * chunk_u4
    res = {}
    mf = cl.mem_flags
    sink = cl.Buffer(ctx, mf.WRITE_ONLY, size=max(n_chunks, 1) * 16)
    for name, off in (("contig", off_c), ("gather", off_g)):
        ob = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=off)
        ts = []
        for r in range(reps):
            ev = prg.read_chunks(q, (n_chunks * lsize,), (lsize,),
                                 src_buf, ob, np.uint32(chunk_u4), sink)
            ev.wait()
            ts.append((ev.profile.end - ev.profile.start) * 1e-9)
        ts = np.array(ts[2:])          # on jette le warmup
        res[name] = (float(np.median(ts)), float(ts.min()))
        ob.release()
    sink.release()
    return res, V_u4 * 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcmb", type=int, default=768)
    ap.add_argument("--volmb", type=int, default=96)
    a = ap.parse_args()

    dev, ctx, q, prg = setup()
    print("=" * 96)
    print(f"GPU : {dev.name}  |  {dev.max_compute_units} CU  |  "
          f"{dev.max_clock_frequency} MHz  |  ligne de cache {dev.global_mem_cacheline_size} o  "
          f"|  L2 {dev.global_mem_cache_size//1024} Kio")
    print(f"pilote {dev.driver_version}  |  {dev.platform.name}")
    print("=" * 96)

    src_u4 = (a.srcmb << 20) // 16
    rng = np.random.default_rng(0)
    host = rng.integers(0, 2**32, size=(src_u4, 4), dtype=np.uint32)
    src_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                        hostbuf=host)
    del host
    print(f"buffer source {a.srcmb} Mio, volume lu par mesure {a.volmb} Mio\n")

    vol_u4 = (a.volmb << 20) // 16
    rows = []
    print(f"{'morceau':>9s} {'= r,D':>10s} {'nb morceaux':>12s} {'etalement':>10s} "
          f"{'contigu GB/s':>13s} {'gather GB/s':>12s} {'ratio':>7s}")
    print("-" * 96)
    for chunk_bytes in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        chunk_u4 = chunk_bytes // 16
        n_chunks = vol_u4 // chunk_u4
        if n_chunks < 64:
            continue
        for spread in (1, 4, 16):
            if n_chunks * spread * chunk_u4 > src_u4:
                continue
            res, vbytes = bench(ctx, q, prg, src_buf, src_u4, chunk_u4,
                                n_chunks, spread, rng)
            bw_c = vbytes / res["contig"][0] / 1e9
            bw_g = vbytes / res["gather"][0] / 1e9
            # equivalence en (r, D) pour un resume fp16
            rD = chunk_bytes // 2
            lbl = f"r*D={rD}"
            print(f"{chunk_bytes:8d}o {lbl:>10s} {n_chunks:12d} {spread:9d}x "
                  f"{bw_c:12.1f} {bw_g:11.1f} {bw_g/bw_c:6.3f}")
            rows.append(dict(chunk=chunk_bytes, n=n_chunks, spread=spread,
                             bw_contig=bw_c, bw_gather=bw_g, ratio=bw_g / bw_c,
                             t_contig=res["contig"][0], t_gather=res["gather"][0]))
    (OUT / "bench_gather.json").write_text(json.dumps(
        dict(device=dev.name, cu=dev.max_compute_units,
             clock=dev.max_clock_frequency, driver=dev.driver_version,
             src_mb=a.srcmb, vol_mb=a.volmb, rows=rows), indent=2))
    print(f"\n-> {OUT/'bench_gather.json'}")


if __name__ == "__main__":
    main()
