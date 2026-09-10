"""
Test E -- reconcilier la contradiction "moyenne vs max" entre notre mesure (mean bat
legerement maxpool, rappel de masse) et celle de l'agent externe (mean perd de 2,2-2,4 nat,
perte de langage) sur le PREMIER ETAGE du selecteur.

Protocole calque au plus pres sur celui de l'agent (harness e2e_qwen.py / echelle_two.py) :
  - taille de bloc Lb = 64
  - fenetre locale W = 8 TOKENS (pas 8 blocs comme notre defaut eval_selection.py)
  - selection m = 1 seul bloc au premier etage (le cas le plus extreme, le plus sensible)
  - score "mean" = Kb.mean(axis=Lb) . q   -- moyenne BRUTE non normalisee, comme le harness
  - score "maxip" = max_j (k_j . q)       -- max sur les cles brutes du bloc
  - metrique : RAPPEL DE MASSE (la notre), pas la perte en nats (celle de l'agent) --
    objectif : voir si le desaccord vient de la metrique ou du modele/protocole.

Gratuit, local, capture reelle Qwen3-8B deja telechargee.
"""
import numpy as np, pathlib, json, collections

F = pathlib.Path(__file__).resolve().parents[1] / "12_poc" / "resultats" / "qk_qwen8b_gpu.npz"
OUT = pathlib.Path(__file__).resolve().parent / "test_e_mean_vs_max.json"

z = np.load(F, allow_pickle=False)
meta = json.loads(str(z["meta"]))
H, KVH, D, T = meta["H"], meta["KVH"], meta["D"], meta["seq"]
layers = meta["layers"]
Lb = 64
W = 8            # fenetre locale en TOKENS, comme le harness (pas en blocs)
s = 1.0 / np.sqrt(D)
g = H // KVH


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


acc = collections.defaultdict(lambda: {k: [] for k in (1, 2, 4, 8)})
n_series = 0
for li in layers:
    for doc in (0, 1):
        kk, qq = f"{doc}_{li}_k_rope", f"{doc}_{li}_q_rope"
        if kk not in z:
            continue
        K_all, Q_all = z[kk], z[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            h = kv * g
            Q = Q_all[:, h, :].astype(np.float32)
            positions = np.arange(256, T - 1, 23)   # echantillon regulier, apres un contexte minimal
            for p in positions:
                hi = (p - W) // Lb          # dernier bloc complet AVANT la fenetre locale
                if hi < 4:
                    continue
                Kb = K[:hi * Lb].reshape(hi, Lb, D)
                q = Q[p]
                # scores exacts par bloc (LSE, l'oracle de masse) pour definir le vrai classement
                exact = np.array([lse(Kb[b] @ q * s) for b in range(hi)])
                mass = np.exp(exact - exact.max()); mass /= mass.sum()
                order_true = np.argsort(-exact)
                # score "mean" (mecanisme du harness, brut non normalise)
                mean_score = (Kb.mean(axis=1) @ q) * s
                # score "maxip" (max sur les cles brutes du bloc)
                maxip_score = (Kb @ q * s).max(axis=1)
                for k in (1, 2, 4, 8):
                    if hi < k:
                        continue
                    top_true = set(order_true[:k].tolist())
                    sel_mean = set(np.argsort(-mean_score)[:k].tolist())
                    sel_max = set(np.argsort(-maxip_score)[:k].tolist())
                    ora_mass = mass[list(top_true)].sum()
                    acc["mean"][k].append(mass[list(sel_mean)].sum() / max(ora_mass, 1e-12))
                    acc["maxip"][k].append(mass[list(sel_max)].sum() / max(ora_mass, 1e-12))
                n_series += 1

print(f"Test E -- premier etage, Lb=64, fenetre locale W=8 TOKENS (protocole du harness)")
print(f"{n_series} series de mesure\n")
print(f"{'k (blocs choisis)':>18s} {'mean (rappel relatif)':>22s} {'maxip (rappel relatif)':>23s} {'ecart':>10s}")
summary = {}
for k in (1, 2, 4, 8):
    mv = np.mean(acc["mean"][k]) if acc["mean"][k] else float("nan")
    xv = np.mean(acc["maxip"][k]) if acc["maxip"][k] else float("nan")
    print(f"{k:18d} {100*mv:21.2f}% {100*xv:22.2f}% {100*(xv-mv):9.2f} pts")
    summary[k] = dict(mean=float(mv), maxip=float(xv))

OUT.write_text(json.dumps(summary, indent=2))
print("\n->", OUT)
