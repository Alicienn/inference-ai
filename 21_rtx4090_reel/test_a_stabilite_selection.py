"""
Test A -- stabilite du sous-ensemble de blocs selectionnes entre pas de decodage consecutifs.

Decide si l'idee "reutiliser la selection tous les k pas" (proposee pour abaisser le seuil
N* de croisement de latence sur RTX4090) est viable : si le top-k de blocs change lentement
d'un pas au suivant, on peut sauter le recalcul de selection pendant k pas sans perdre de
masse -- ce qui diviserait le plateau de cout du selecteur par k.

Gratuit : utilise la capture Q/K reelle de Qwen3-8B deja telechargee (pas de GPU).
Selecteur : max inner-product par bloc (le score corrige du papier, meilleur que la moyenne).
"""
import numpy as np, pathlib, json, collections

F = pathlib.Path(__file__).resolve().parents[1] / "12_poc" / "resultats" / "qk_qwen8b_gpu.npz"
OUT = pathlib.Path(__file__).resolve().parent / "test_a_stabilite_selection.json"

z = np.load(F, allow_pickle=False)
meta = json.loads(str(z["meta"]))
H, KVH, D, T = meta["H"], meta["KVH"], meta["D"], meta["seq"]
layers = meta["layers"]
Lb = 64
K_TOPS = (8, 32)          # tailles de selection testees (~ kfrac=64 et mfrac=8 de e2e_qwen.py)
GAPS = (1, 2, 4, 8, 16, 32)
s = 1.0 / np.sqrt(D)
g = H // KVH


def scores_maxip(Kb, q):
    return (Kb @ q * s).max(axis=1)          # (n,) max sur les Lb cles du bloc


results = {}
for li in layers:
    for doc in (0, 1):
        kk, qq = f"{doc}_{li}_k_rope", f"{doc}_{li}_q_rope"
        if kk not in z:
            continue
        K_all, Q_all = z[kk], z[qq]           # (T,KVH,D) / (T,H,D)
        n = T // Lb
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            Kb = K[:n * Lb].reshape(n, Lb, D)
            h = kv * g                        # une tete par groupe suffit pour ce diagnostic
            Q = Q_all[:, h, :].astype(np.float32)
            positions = np.arange(int(T * 0.5), T - max(GAPS) - 1, 37)
            for k in K_TOPS:
                overlaps = collections.defaultdict(list)
                for p in positions:
                    hi = p // Lb
                    if hi <= k + 2:
                        continue
                    top0 = set(np.argsort(-scores_maxip(Kb[:hi], Q[p]))[:k].tolist())
                    for gap in GAPS:
                        p2 = p + gap
                        if p2 >= T:
                            continue
                        hi2 = p2 // Lb
                        top2 = set(np.argsort(-scores_maxip(Kb[:hi2], Q[p2]))[:k].tolist())
                        overlaps[gap].append(len(top0 & top2) / len(top0 | top2))
                key = f"L{li}_kv{kv}_k{k}"
                results[key] = {str(gp): float(np.mean(v)) for gp, v in overlaps.items() if v}

agg = collections.defaultdict(list)
for v in results.values():
    for gp, val in v.items():
        agg[gp].append(val)
summary = {gp: float(np.mean(vals)) for gp, vals in sorted(agg.items(), key=lambda x: int(x[0]))}

print("Chevauchement (Jaccard) du top-k de blocs selectionnes, pas p vs p+gap :")
print(f"{'gap (pas)':>10s} {'jaccard moyen':>14s}   lecture")
for gp, v in summary.items():
    lecture = "quasi identique -> reutilisable" if v > 0.85 else (
        "change vite -> pas reutilisable" if v < 0.5 else "intermediaire")
    print(f"{gp:>10s} {v:>14.3f}   {lecture}")

OUT.write_text(json.dumps({"summary": summary, "n_series": len(results)}, indent=2))
print("\n->", OUT)
