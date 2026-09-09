# -*- coding: utf-8 -*-
"""Position 0 (BOS) seule, ou toutes les positions = 0 mod 64 ?
Reconciliation : masse locale + masse candidate + masse oracle."""
import numpy as np, json, pathlib
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")


def sma(S):
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(1, keepdims=True)


for tag in ("smol8k", "qwen8k"):
    d = np.load(P / f"qk_{tag}.npz", mmap_mode="r")
    meta = json.loads(str(d["meta"]))
    H, KVH, D, layers = meta["H"], meta["KVH"], meta["D"], meta["layers"]
    grp = H // KVH
    m0, mr0, mloc, mtop = [], [], [], []
    top_pos = np.zeros(64)
    for layer in layers:
        K = np.asarray(d[f"0_{layer}_k_rope"]).astype(np.float64)
        Q = np.asarray(d[f"0_{layer}_q_rope"]).astype(np.float64)
        T = K.shape[0]
        for p in (T - 8, T - 1):
            for kv in range(KVH):
                qg = Q[p, kv * grp:(kv + 1) * grp]
                Kc = K[:p + 1, kv]
                P_ = sma((qg @ Kc.T) / np.sqrt(D))
                loc = np.zeros(p + 1, dtype=bool); loc[max(0, p - 512 + 1):p + 1] = True
                nbc = max(0, (p - 511) // 64)
                if nbc < 8:
                    continue
                r0 = np.zeros(p + 1, dtype=bool); r0[::64] = True
                for gi in range(len(qg)):
                    m0.append(P_[gi, 0])
                    mr0.append(P_[gi, r0].sum())
                    mloc.append(P_[gi, loc].sum())
                    tp = np.argsort(-P_[gi])[:8]
                    mtop.append(P_[gi, tp].sum())
                    for t in tp:
                        top_pos[int(t) % 64] += 1
    print(f"\n=== {tag} (par tete-requete, {len(m0)} echantillons) ===")
    print(f"  masse position 0 seule        : {np.mean(m0):.4f} (med {np.median(m0):.4f})")
    print(f"  masse classe r=0 (mod 64)     : {np.mean(mr0):.4f}")
    print(f"  -> part de la position 0 dans la classe r=0 : {100*np.mean(m0)/np.mean(mr0):.1f} %")
    print(f"  masse fenetre locale 512      : {np.mean(mloc):.4f}")
    print(f"  masse des 8 positions de tete : {np.mean(mtop):.4f}")
    tp = np.argsort(-top_pos)[:8]
    print(f"  positions les plus souvent dans le top-8 (mod 64) : " +
          ", ".join(f"r={r} ({100*top_pos[r]/top_pos.sum():.1f} %)" for r in tp))
