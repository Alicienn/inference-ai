# -*- coding: utf-8 -*-
"""Robustesse du selecteur par cles reelles : le resultat depend-il du choix d'offset ?
Schemas : s=1 aux offsets 0/16/32/48 ; s=1 offset aleatoire par bloc ; s=2 et s=4 etales ;
s=4 aleatoires ; oracle (s=64). Passe 2 = 4 bits, W=512, m=8, causal, toutes les tetes."""
import numpy as np, json, pathlib
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats")
NEG = -1e30


def sma(S):
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=1, keepdims=True)


def quant_b(Kb, bits):
    mn = Kb.min(axis=1, keepdims=True); mx = Kb.max(axis=1, keepdims=True)
    step = np.maximum(mx - mn, 1e-8) / (2 ** bits - 1)
    return np.round((Kb - mn) / step) * step + mn


SCHEMES = [("s1_off0", "fixe", 1, 0), ("s1_off16", "fixe", 1, 16), ("s1_off32", "fixe", 1, 32),
           ("s1_off48", "fixe", 1, 48), ("s1_rand", "rand", 1, 0), ("s2_etal", "etal", 2, 0),
           ("s4_etal", "etal", 4, 0), ("s4_rand", "rand", 4, 0), ("oracle", "full", 64, 0)]


def sweep(tag, doc=0, Lb=64, W=512, nq=8, m=8, bits=4, seed=0):
    d = np.load(P / f"qk_{tag}.npz", mmap_mode="r")
    meta = json.loads(str(d["meta"]))
    H, KVH, D, layers = meta["H"], meta["KVH"], meta["D"], meta["layers"]
    grp = H // KVH
    rng = np.random.default_rng(seed)
    rows = []
    for layer in layers:
        K = np.asarray(d[f"{doc}_{layer}_k_rope"]).astype(np.float64)
        Q = np.asarray(d[f"{doc}_{layer}_q_rope"]).astype(np.float64)
        T = K.shape[0]
        for p in range(T - nq, T):
            for kv in range(KVH):
                qg = Q[p, kv * grp:(kv + 1) * grp]
                Kc = K[:p + 1, kv]
                S = (qg @ Kc.T) / np.sqrt(D)
                P_ = sma(S)
                O = P_ @ Kc
                nO = np.linalg.norm(O, axis=1)
                loc = np.zeros(p + 1, dtype=bool); loc[max(0, p - W + 1):p + 1] = True
                nbc = max(0, (p - W + 1) // Lb)
                if nbc < m:
                    continue
                Kb = Kc[:nbc * Lb].reshape(nbc, Lb, D)
                Kq = quant_b(Kb, bits).reshape(nbc * Lb, D)
                Kfull = np.vstack([Kq, Kc[nbc * Lb:]])
                sq = (qg @ Kfull.T) / np.sqrt(D)
                Sb = S[:, :nbc * Lb].reshape(len(qg), nbc, Lb)
                base = len(rows)
                for gi in range(len(qg)):
                    rows.append({"tag": tag, "layer": int(layer), "head": kv * grp + gi})
                for name, kind, s, off in SCHEMES:
                    if kind == "full":
                        sc = Sb
                    elif kind == "fixe":
                        idx = (np.arange(s) * (Lb // s) + off) % Lb
                        sc = Sb[:, :, idx]
                    elif kind == "etal":
                        idx = (np.arange(s) * (Lb // s)).astype(int)
                        sc = Sb[:, :, idx]
                    else:  # rand : un offset aleatoire par bloc
                        idx = rng.integers(0, Lb, size=(nbc, s))
                        sc = np.take_along_axis(Sb, np.broadcast_to(idx, (len(qg), nbc, s)), axis=2)
                    mx = sc.max(2, keepdims=True)
                    lse = np.log(np.exp(sc - mx).sum(2)) + mx[:, :, 0]
                    tm = np.argsort(-lse, axis=1)[:, :m]
                    kk = np.tile(loc, (len(qg), 1))
                    for gi in range(len(qg)):
                        for b in tm[gi]:
                            kk[gi, int(b) * Lb:(int(b) + 1) * Lb] = True
                    mass = (P_ * kk).sum(1)
                    err = np.linalg.norm(sma(np.where(kk, sq, NEG)) @ Kc - O, axis=1) / nO
                    for gi in range(len(qg)):
                        rows[base + gi][f"mass_{name}"] = float(mass[gi])
                        rows[base + gi][f"err_{name}"] = float(err[gi])
    return rows


if __name__ == "__main__":
    allrows = []
    for tag in ("smol8k", "qwen8k"):
        r = sweep(tag)
        allrows += r
        print(f"{tag}: {len(r)} lignes", flush=True)
    (OUT / "pass1_offsets.json").write_text(json.dumps(allrows), encoding="utf-8")
    print("\n=== robustesse du schema d'echantillonnage (par tete, moyenne 8 requetes) ===")
    for tag in ("smol8k", "qwen8k"):
        sel = [x for x in allrows if x["tag"] == tag]
        heads = {}
        for x in sel:
            heads.setdefault((x["layer"], x["head"]), []).append(x)
        print(f"\n{tag} : {len(heads)} tetes")
        for name, _, _, _ in SCHEMES:
            e = np.array([np.mean([y[f"err_{name}"] for y in v]) for v in heads.values()])
            ms = np.array([np.mean([y[f"mass_{name}"] for y in v]) for v in heads.values()])
            print(f"   {name:10s} err med={np.median(e):.4f} p90={np.percentile(e,90):.4f} "
                  f"max={e.max():.4f} | catastrophes={100*(e>0.5).mean():5.1f} % | masse med={np.median(ms):.3f}")
