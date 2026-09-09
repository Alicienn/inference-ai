"""
PISTE 3 — proxy CPU.  [SIGNAL FAIBLE — NON TRANSPOSABLE AU GPU]

Ce test ne mesure NI la coalescence de warp NI le comportement du GPU. Il mesure la
sensibilite d'une hierarchie de caches CPU (L1/L2/L3 + prefetcher) a un motif d'acces
disperse par morceaux, a volume identique. Il ne sert que de contre-point sur une
architecture memoire differente. Aucune conclusion sur ASP n'en decoule.
"""
import numpy as np, time, json, pathlib

def bench(src, chunks_u8, order, reps=7):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        acc = np.uint64(0)
        for o in order:
            acc ^= np.bitwise_xor.reduce(src[o:o + chunks_u8])
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

SRC_MB, VOL_MB = 512, 64
src = np.random.default_rng(0).integers(0, 2**63, size=(SRC_MB << 20)//8, dtype=np.uint64)
rng = np.random.default_rng(1)
print("="*88)
print("PROXY CPU  [SIGNAL FAIBLE - non transposable au GPU]")
print("="*88)
print(f"{'morceau':>9s} {'nb':>8s} {'contigu GB/s':>13s} {'gather GB/s':>12s} {'ratio':>7s}")
print("-"*88)
rows=[]
for cb in (256, 1024, 4096, 16384, 65536):
    cu = cb//8; n = (VOL_MB<<20)//cb
    if n < 16: continue
    oc = np.arange(n, dtype=np.int64)*cu
    og = rng.choice(len(src)//cu, size=n, replace=False).astype(np.int64)*cu
    tc, tg = bench(src, cu, oc), bench(src, cu, og)
    v = n*cb
    print(f"{cb:8d}o {n:8d} {v/tc/1e9:12.2f} {v/tg/1e9:11.2f} {tc/tg:6.3f}")
    rows.append(dict(chunk=cb, bw_contig=v/tc/1e9, bw_gather=v/tg/1e9, ratio=tc/tg))
pathlib.Path("resultats/proxy_cpu.json").write_text(json.dumps(rows, indent=2))
print("""
[SIGNAL FAIBLE] Le CPU montre la meme tendance qualitative que le GPU : la penalite
du gather s'attenue quand la taille de morceau croit. Mais les mecanismes different
(prefetcher materiel et lignes de 64 o cote CPU ; coalescence de wavefront et pages
DRAM cote GPU). Ce resultat ne vaut PAS validation, seulement contre-point.""")
