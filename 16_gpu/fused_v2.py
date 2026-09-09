"""
NOYAU ASP FUSIONNE — v2, apres correction de deux artefacts de banc.

CORRECTIONS PAR RAPPORT A LA v1
-------------------------------
(1) COMPARAISON A CHARGE EGALE. Dans un vrai pas de decodage, chaque requete
    (batch x tetes KV) doit scanner TOUS les n blocs : le travail du selecteur vaut
    Q*n*S1, pas n*S1. La v1 faisait lire les resumes une seule fois aux chemins plat et
    2-noyaux, mais une fois par requete au fusionne -> comparaison invalide d'un facteur Q.
    Les trois chemins font desormais Q requetes.

(2) COALESCENCE EN PHASE 1. La v1 faisait lire un resume de bloc entier par UN thread,
    avec des adresses distantes de G*c1 entre threads voisins : non coalesce. Desormais
    une EQUIPE de TW=32 threads lit un resume ensemble (acces contigus), avec reduction
    intra-equipe en memoire partagee. Meme classe d'erreur que l'artefact de la passe 1
    en session 4, transposee au niveau du thread.

TROIS CHEMINS, MEME VOLUME LOGIQUE, MEME Q :
  plat     : Q * n resumes fins, flux contigu                         (1 lancement)
  2-noyaux : Q * n resumes grossiers (flux) puis Q * m gathers fins   (2 lancements)
  fusionne : idem, en 1 lancement, selection stratifiee entrelacee
"""
import numpy as np, pyopencl as cl, json, pathlib, argparse

OUT = pathlib.Path(__file__).resolve().parent / "resultats"
OUT.mkdir(exist_ok=True)
LS, TW, QMAX = 256, 32, 32
NT = LS // TW

SRC = r"""
#define LS 256
#define TW 32
#define NT (LS/TW)
#define QMAX 32
inline uint h4(uint4 v){ return v.x ^ (v.y*2654435761u) ^ (v.z<<7) ^ (v.w>>3); }

/* flux contigu : sert au chemin plat ET a la passe 1 du 2-noyaux */
__kernel void stream(__global const uint4* src, const uint span, __global uint* sink){
    const uint g=get_group_id(0), lid=get_local_id(0);
    uint acc=0; const uint base=g*span;
    for(uint i=lid;i<span;i+=LS) acc ^= h4(src[base+i]);
    __local uint red[LS]; red[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);
    for(uint s=LS>>1;s;s>>=1){ if(lid<s) red[lid]^=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE);}
    if(lid==0) sink[g]=red[0];
}

/* gather : passe 2 du 2-noyaux. Un work-group par morceau. */
__kernel void gather(__global const uint4* src, __global const uint* off,
                     const uint chunk, __global uint* sink){
    const uint g=get_group_id(0), lid=get_local_id(0);
    uint acc=0; const uint base=off[g];
    for(uint i=lid;i<chunk;i+=LS) acc ^= h4(src[base+i]);
    __local uint red[LS]; red[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);
    for(uint s=LS>>1;s;s>>=1){ if(lid<s) red[lid]^=red[lid+s]; barrier(CLK_LOCAL_MEM_FENCE);}
    if(lid==0) sink[g]=red[0];
}

/* FUSIONNE. grille = Q*G groupes ; gid%G = strate entrelacee, gid/G = requete.
   Phase 1 : NT equipes de TW threads, chaque equipe lit un resume grossier de facon
   COALESCEE, reduit, et garde son meilleur bloc. Phase 2 : gather des q selectionnes. */
__kernel void fused(__global const uint4* coarse, __global const uint4* fine,
                    const uint n, const uint G, const uint c1, const uint c2,
                    const uint q, __global uint* sink)
{
    const uint gid=get_group_id(0), lid=get_local_id(0);
    const uint strate = gid % G;
    const uint team = lid / TW, lane = lid % TW;

    __local uint part[LS];              /* reduction intra-equipe */
    __local uint bs[NT*QMAX], bi[NT*QMAX], thr[NT];   /* top-q par equipe + seuil */
    for(uint t=lane; t<q; t+=TW){ bs[team*QMAX+t]=0u; bi[team*QMAX+t]=0u; }
    if(lane==0) thr[team]=0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint b0=strate; b0<n; b0+=G*NT){
        const uint b = b0 + team*G;
        uint acc=0;
        if(b<n){ const uint base=b*c1;
                 for(uint i=lane;i<c1;i+=TW) acc ^= h4(coarse[base+i]); }
        part[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);
        for(uint s=TW>>1;s;s>>=1){
            if(lane<s) part[lid]^=part[lid+s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        /* Insertion a SEUIL : le cas courant (score sous le q-ieme meilleur) coute O(1).
           Sans ce test, l'insertion en O(q) s'executait sur UNE voie a chaque iteration
           pendant que 255 threads attendaient la barriere -> goulot serie mesure. */
        if(lane==0 && b<n){
            uint sc=part[team*TW];
            if(sc > thr[team]){
                uint si=b;
                for(uint t=0;t<q;t++){
                    if(sc>bs[team*QMAX+t]){
                        uint ts=bs[team*QMAX+t], ti=bi[team*QMAX+t];
                        bs[team*QMAX+t]=sc; bi[team*QMAX+t]=si; sc=ts; si=ti;
                    }
                }
                thr[team]=bs[team*QMAX+q-1];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    /* fusion des NT tops locaux en un top-q du work-group */
    __local uint sel[QMAX];
    if(lid==0){
        for(uint t=0;t<q;t++){
            uint best=0,bidx=0,bj=0;
            for(uint j=0;j<NT*QMAX;j++) if(bs[j]>best){ best=bs[j]; bidx=bi[j]; bj=j; }
            sel[t]=bidx; bs[bj]=0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* phase 2 : gather coalesce des q resumes fins */
    uint acc=0;
    for(uint t=0;t<q;t++){
        const uint base=sel[t]*c2;
        for(uint i=lid;i<c2;i+=LS) acc ^= h4(fine[base+i]);
    }
    part[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);
    for(uint s=LS>>1;s;s>>=1){ if(lid<s) part[lid]^=part[lid+s]; barrier(CLK_LOCAL_MEM_FENCE);}
    if(lid==0) sink[gid]=part[0];
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--Q", type=int, default=16, help="requetes (batch x tetes KV)")
    a = ap.parse_args()
    dev = next(d for p in cl.get_platforms() for d in p.get_devices()
               if d.type & cl.device_type.GPU)
    ctx = cl.Context([dev])
    cq = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prg = cl.Program(ctx, SRC).build(options="-cl-std=CL2.0")
    kst, kga, kfu = (cl.Kernel(prg, x) for x in ("stream", "gather", "fused"))
    mf = cl.mem_flags
    print("=" * 108)
    print(f"ASP FUSIONNE v2  --  {dev.name}, {dev.max_compute_units} CU  --  Q={a.Q} requetes")
    print("  (les trois chemins font Q requetes ; phase 1 coalescee par equipes de 32)")
    print("=" * 108)
    rng = np.random.default_rng(0)
    su4 = (512 << 20) // 16
    host = rng.integers(0, 2 ** 32, size=(su4, 4), dtype=np.uint32)
    buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host); del host
    sink = cl.Buffer(ctx, mf.WRITE_ONLY, size=(1 << 23))

    def run_k(k, gsize, args, reps):
        ts = []
        for _ in range(reps):
            k.set_args(*args)
            ev = cl.enqueue_nd_range_kernel(cq, k, (gsize * LS,), (LS,)); ev.wait()
            ts.append((ev.profile.end - ev.profile.start) * 1e-9)
        return float(np.median(ts[2:]))

    rows = []
    print(f"\n{'n':>6s} {'D':>4s} {'r1':>3s} {'r2':>3s} {'m':>5s} {'G':>4s} {'q':>3s} | "
          f"{'plat ms':>9s} {'2noyaux':>9s} {'fusionne':>9s} | {'gain 2k':>8s} "
          f"{'gain fus':>9s} {'fus/2k':>8s}")
    print("-" * 108)
    for D in (64, 128):
        for n in (4096, 16384):
            for r1, r2 in ((2, 8), (2, 16), (4, 16)):
                S1, S2 = r1 * D * 2, r2 * D * 2
                c1, c2 = S1 // 16, S2 // 16
                if c1 < 1 or c2 < 1 or n * max(c1, c2) > su4:
                    continue
                for frac in (4, 8):
                    m = n // frac
                    G = 16
                    q = min(max(m // G, 1), QMAX)
                    # --- plat : Q * n resumes fins, contigu
                    tf = a.Q * run_k(kst, max(n * S2 // 16384, 1),
                                     (buf, np.uint32(16384 // 16), sink), a.reps)
                    # --- 2 noyaux : Q * (flux grossier) + Q * (gather fin)
                    t1 = a.Q * run_k(kst, max(n * S1 // 16384, 1),
                                     (buf, np.uint32(16384 // 16), sink), a.reps)
                    pick = (rng.choice(n, size=m, replace=False).astype(np.uint32) * c2)
                    ob = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pick)
                    t2 = a.Q * run_k(kga, m, (buf, ob, np.uint32(c2), sink), a.reps)
                    ob.release()
                    # --- fusionne : Q*G groupes, un seul lancement
                    tfu = run_k(kfu, a.Q * G,
                                (buf, buf, np.uint32(n), np.uint32(G), np.uint32(c1),
                                 np.uint32(c2), np.uint32(q), sink), a.reps)
                    g2k, gfu = tf / (t1 + t2), tf / tfu
                    print(f"{n:6d} {D:4d} {r1:3d} {r2:3d} {m:5d} {G:4d} {q:3d} | "
                          f"{1e3*tf:9.4f} {1e3*(t1+t2):9.4f} {1e3*tfu:9.4f} | "
                          f"{g2k:7.2f}x {gfu:8.2f}x {(t1+t2)/tfu:7.2f}x")
                    rows.append(dict(n=n, D=D, r1=r1, r2=r2, m=m, G=G, q=q, Q=a.Q,
                                     t_flat=tf, t_2k=t1 + t2, t_fused=tfu,
                                     gain_2k=g2k, gain_fused=gfu,
                                     speedup_fusion=(t1 + t2) / tfu))
    sp = np.array([r["speedup_fusion"] for r in rows])
    gf = np.array([r["gain_fused"] for r in rows])
    g2 = np.array([r["gain_2k"] for r in rows])
    print("-" * 108)
    print(f"\n  gain vs plat  : 2 noyaux mediane {np.median(g2):.2f}x   "
          f"fusionne mediane {np.median(gf):.2f}x")
    print(f"  fusion vs 2 noyaux : mediane {np.median(sp):.2f}x  "
          f"[{sp.min():.2f} - {sp.max():.2f}]   "
          f"({int((sp<1).sum())}/{len(sp)} ou la fusion perd)")
    (OUT / "fused_v2.json").write_text(json.dumps(dict(device=dev.name, Q=a.Q, rows=rows), indent=2))
    print(f"\n-> {OUT/'fused_v2.json'}")


if __name__ == "__main__":
    main()
