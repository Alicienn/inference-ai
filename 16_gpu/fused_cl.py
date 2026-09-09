"""
NOYAU ASP FUSIONNE — version OpenCL, validee sur gfx1152 avant portage CUDA.

CONCEPTION (cf. journal session 8, conception C).
Un seul lancement. Grille = Q requetes x G work-groups par requete. Chaque work-group
traite une STRATE ENTRELACEE des n blocs (indices gi, gi+G, gi+2G, ...), en garde le
top-q localement (q = m/G), puis gathere immediatement ses propres survivants.
ZERO synchronisation, parallelisme complet, un seul lancement.

Le compromis — sélection stratifiee au lieu du top-m global — a ete quantifie AVANT
d'ecrire ce code (12_poc/code/stratified.py) : perte maximale de 2,9 a 3,2 points
relatifs en mode entrelace, contre jusqu'a 22,3 en mode contigu. L'entrelacement est
donc obligatoire, et c'est aussi le motif naturel d'une boucle a pas de grille.

Second niveau de stratification, interne au work-group : chaque thread garde son top-2
en registres, puis le groupe fusionne LS*2 candidats en top-q. Meme argument
d'echangeabilite ; quota 2 par thread pour rester au-dessus du regime a quota 1.

TROIS CHEMINS MESURES DANS LE MEME BINAIRE, a volume identique :
  plat    : flux contigu des n resumes fins            (1 lancement)
  asp2k   : flux contigu des n resumes grossiers, puis gather de m fins (2 lancements)
  fusionne: le tout en 1 lancement, selection stratifiee
"""
import numpy as np, pyopencl as cl, json, pathlib, argparse

OUT = pathlib.Path(__file__).resolve().parent / "resultats"
OUT.mkdir(exist_ok=True)

SRC = r"""
#define LS 128
#define TPT 2            /* top-2 par thread */
#define QMAX 16

inline uint h4(uint4 v){ return v.x ^ (v.y*2654435761u) ^ (v.z<<7) ^ (v.w>>3); }

/* ---- flux contigu : reference (schema plat, et passe 1 du 2-noyaux) ---- */
__kernel void stream(__global const uint4* src, const uint span, __global uint* sink){
    const uint g=get_group_id(0), lid=get_local_id(0);
    uint acc=0; const uint base=g*span;
    for(uint i=lid;i<span;i+=LS) acc ^= h4(src[base+i]);
    __local uint red[LS]; red[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);
    for(uint s=LS>>1;s;s>>=1){ if(lid<s) red[lid]^=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE);}
    if(lid==0) sink[g]=red[0];
}

/* ---- gather : passe 2 du schema a deux noyaux ---- */
__kernel void gather(__global const uint4* src, __global const uint* off,
                     const uint chunk, __global uint* sink){
    const uint g=get_group_id(0), lid=get_local_id(0);
    uint acc=0; const uint base=off[g];
    for(uint i=lid;i<chunk;i+=LS) acc ^= h4(src[base+i]);
    __local uint red[LS]; red[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);
    for(uint s=LS>>1;s;s>>=1){ if(lid<s) red[lid]^=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE);}
    if(lid==0) sink[g]=red[0];
}

/* ---- FUSIONNE : selection stratifiee + gather, un seul lancement ----
   grille : Q*G groupes.  gid%G = strate entrelacee.
   n blocs, resume grossier de c1 uint4, resume fin de c2 uint4, quota q par strate. */
__kernel void fused(__global const uint4* coarse, __global const uint4* fine,
                    const uint n, const uint G, const uint c1, const uint c2,
                    const uint q, __global uint* sink)
{
    const uint gid=get_group_id(0), lid=get_local_id(0);
    const uint strate = gid % G;

    /* --- phase 1 : score de la strate entrelacee, top-TPT par thread (registres) --- */
    uint bs[TPT], bi[TPT];
    #pragma unroll
    for(int t=0;t<TPT;t++){ bs[t]=0u; bi[t]=0u; }
    for(uint b=strate+lid*G; b<n; b+=G*LS){
        uint acc=0; const uint base=b*c1;
        for(uint i=0;i<c1;i++) acc ^= h4(coarse[base+i]);
        uint sc=acc, si=b;                         /* insertion par decalage */
        #pragma unroll
        for(int t=0;t<TPT;t++){
            if(sc>bs[t]){ uint ts=bs[t], ti=bi[t]; bs[t]=sc; bi[t]=si; sc=ts; si=ti; }
        }
    }
    /* --- fusion des LS*TPT candidats en top-q --- */
    __local uint cs[LS*TPT], ci[LS*TPT], selidx[QMAX];
    #pragma unroll
    for(int t=0;t<TPT;t++){ cs[lid*TPT+t]=bs[t]; ci[lid*TPT+t]=bi[t]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid==0){
        /* q passes d'extraction du maximum, marquage par mise a zero.
           q<=16 sur LS*TPT=256 candidats : ~4096 operations sur un thread,
           negligeable devant les lectures memoire (verifie par ventilation). */
        for(uint t=0;t<q;t++){
            uint best=0, bidx=0, bj=0;
            for(uint j=0;j<LS*TPT;j++){ if(cs[j]>best){ best=cs[j]; bidx=ci[j]; bj=j; } }
            selidx[t]=bidx; cs[bj]=0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    /* --- phase 2 : gather des q resumes fins selectionnes --- */
    uint acc=0;
    for(uint t=0;t<q;t++){
        const uint base=selidx[t]*c2;
        for(uint i=lid;i<c2;i+=LS) acc ^= h4(fine[base+i]);
    }
    __local uint red[LS]; red[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);
    for(uint s=LS>>1;s;s>>=1){ if(lid<s) red[lid]^=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE);}
    if(lid==0) sink[gid]=red[0];
}
"""
LS = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=11)
    a = ap.parse_args()
    dev = next(d for p in cl.get_platforms() for d in p.get_devices()
               if d.type & cl.device_type.GPU)
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prg = cl.Program(ctx, SRC).build(options="-cl-std=CL2.0")
    kst, kga, kfu = (cl.Kernel(prg, x) for x in ("stream", "gather", "fused"))
    mf = cl.mem_flags
    print("=" * 104)
    print(f"ASP FUSIONNE vs 2 NOYAUX  --  {dev.name}, {dev.max_compute_units} CU, "
          f"{dev.max_clock_frequency} MHz")
    print("=" * 104)

    rng = np.random.default_rng(0)
    SRC_MB = 512
    su4 = (SRC_MB << 20) // 16
    host = rng.integers(0, 2 ** 32, size=(su4, 4), dtype=np.uint32)
    buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host); del host
    sink = cl.Buffer(ctx, mf.WRITE_ONLY, size=(1 << 22))

    def t_stream(nbytes, reps):
        span = 16384 // 16
        ng = max(nbytes // 16384, 1)
        ts = []
        for _ in range(reps):
            kst.set_args(buf, np.uint32(span), sink)
            ev = cl.enqueue_nd_range_kernel(q, kst, (ng * LS,), (LS,)); ev.wait()
            ts.append((ev.profile.end - ev.profile.start) * 1e-9)
        return float(np.median(ts[2:]))

    def t_gather(offb, c2, ng, reps):
        ts = []
        for _ in range(reps):
            kga.set_args(buf, offb, np.uint32(c2), sink)
            ev = cl.enqueue_nd_range_kernel(q, kga, (ng * LS,), (LS,)); ev.wait()
            ts.append((ev.profile.end - ev.profile.start) * 1e-9)
        return float(np.median(ts[2:]))

    def t_fused(n, G, c1, c2, qq, Q, reps):
        ts = []
        ng = Q * G
        for _ in range(reps):
            kfu.set_args(buf, buf, np.uint32(n), np.uint32(G), np.uint32(c1),
                         np.uint32(c2), np.uint32(qq), sink)
            ev = cl.enqueue_nd_range_kernel(q, kfu, (ng * LS,), (LS,)); ev.wait()
            ts.append((ev.profile.end - ev.profile.start) * 1e-9)
        return float(np.median(ts[2:]))

    rows = []
    print(f"\n{'n':>6s} {'D':>4s} {'r1':>3s} {'r2':>3s} {'m':>6s} {'Q':>4s} {'G':>4s} | "
          f"{'plat ms':>9s} {'2noyaux':>9s} {'fusionne':>9s} | "
          f"{'gain 2k':>8s} {'gain fus':>9s} {'fus/2k':>8s}")
    print("-" * 104)
    for D in (64, 128):
        for n in (4096, 16384):
            for r1, r2 in ((2, 8), (2, 16), (4, 16)):
                S1, S2 = r1 * D * 2, r2 * D * 2
                c1, c2 = S1 // 16, S2 // 16
                if c1 < 1 or c2 < 1 or n * max(c1, c2) > su4:
                    continue
                for frac in (4, 8):
                    m = n // frac
                    Q, G = 64, 8                      # 64 requetes, 8 strates
                    qq = max(m // (Q * 0 + G), 1)
                    if qq > 16:
                        qq = 16
                    by_flat, by1, by2 = n * S2, n * S1, m * S2
                    tf = t_stream(by_flat, a.reps)
                    t1 = t_stream(by1, a.reps)
                    pick = rng.choice(n, size=m, replace=False).astype(np.uint32) * c2
                    ob = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pick)
                    t2 = t_gather(ob, c2, m, a.reps); ob.release()
                    tfu = t_fused(n, G, c1, c2, qq, Q, a.reps)
                    g2k, gfu = tf / (t1 + t2), tf / tfu
                    print(f"{n:6d} {D:4d} {r1:3d} {r2:3d} {m:6d} {Q:4d} {G:4d} | "
                          f"{1e3*tf:9.4f} {1e3*(t1+t2):9.4f} {1e3*tfu:9.4f} | "
                          f"{g2k:7.2f}x {gfu:8.2f}x {(t1+t2)/tfu:7.2f}x")
                    rows.append(dict(n=n, D=D, r1=r1, r2=r2, m=m, Q=Q, G=G, q=qq,
                                     t_flat=tf, t_2k=t1 + t2, t_fused=tfu,
                                     gain_2k=g2k, gain_fused=gfu,
                                     speedup_fusion=(t1 + t2) / tfu))
    sp = np.array([r["speedup_fusion"] for r in rows])
    print("-" * 104)
    print(f"\n  acceleration de la fusion vs 2 noyaux : mediane {np.median(sp):.2f}x  "
          f"[{sp.min():.2f} - {sp.max():.2f}]")
    print(f"  configurations ou la fusion est plus LENTE : "
          f"{int((sp < 1).sum())}/{len(sp)}")
    (OUT / "fused_cl.json").write_text(json.dumps(dict(device=dev.name, rows=rows), indent=2))
    print(f"\n-> {OUT/'fused_cl.json'}")


if __name__ == "__main__":
    main()
