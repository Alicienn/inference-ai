# -*- coding: utf-8 -*-
"""La passe 1 doit-elle lire un RESUME ou quelques CLES REELLES ?
Courbe erreur de sortie vs volume lu par la passe grossiere : s cles echantillonnees par
bloc (s = 1..32), score = log-sum-exp des s scores reels ; s = 64 = oracle (LSE exact).
Passe 2 = 4 bits, fenetre locale W=512, m=8, causal, toutes les tetes."""
import numpy as np, json, pathlib
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats")
NEG = -1e30
S_LIST = (1, 2, 4, 8, 16, 32, 64)


def sma(S):
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=1, keepdims=True)


def quant_b(Kb, bits):
    mn = Kb.min(axis=1, keepdims=True); mx = Kb.max(axis=1, keepdims=True)
    step = np.maximum(mx - mn, 1e-8) / (2 ** bits - 1)
    return np.round((Kb - mn) / step) * step + mn


def sweep(tag, doc=0, Lb=64, W=512, nq=8, m=8, bits=4):
    d = np.load(P / f"qk_{tag}.npz", mmap_mode="r")
    meta = json.loads(str(d["meta"]))
    H, KVH, D, layers = meta["H"], meta["KVH"], meta["D"], meta["layers"]
    grp = H // KVH
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
                    rows.append({"tag": tag, "layer": int(layer), "head": kv * grp + gi,
                                 "masse_locale": float(P_[gi, loc].sum())})
                for s in S_LIST:
                    if s == Lb:
                        sc = Sb
                    else:
                        idx = (np.arange(s) * (Lb // s)).astype(int)
                        sc = Sb[:, :, idx]
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
                        rows[base + gi][f"mass_s{s}"] = float(mass[gi])
                        rows[base + gi][f"err_s{s}"] = float(err[gi])
    return rows


if __name__ == "__main__":
    allrows = []
    for tag in ("smol8k", "qwen8k"):
        r = sweep(tag)
        allrows += r
        print(f"{tag}: {len(r)} lignes", flush=True)
    (OUT / "pass1_sampled.json").write_text(json.dumps(allrows), encoding="utf-8")
    print("\n=== erreur de sortie vs volume lu par la passe 1 (par tete, moyenne 8 requetes) ===")
    print("   (octets lus par requete = 128 blocs x s cles x 64 dims x 2 o)")
    for tag in ("smol8k", "qwen8k"):
        sel = [x for x in allrows if x["tag"] == tag]
        heads = {}
        for x in sel:
            heads.setdefault((x["layer"], x["head"]), []).append(x)
        print(f"\n{tag} : {len(heads)} tetes")
        for s in S_LIST:
            e = np.array([np.mean([y[f"err_s{s}"] for y in v]) for v in heads.values()])
            ms = np.array([np.mean([y[f"mass_s{s}"] for y in v]) for v in heads.values()])
            print(f"   s={s:2d} ({128*s*64*2//1024:5d} Ko) err med={np.median(e):.4f} "
                  f"p90={np.percentile(e,90):.4f} max={e.max():.4f} | "
                  f"catastrophes={100*(e>0.5).mean():5.1f} % | masse med={np.median(ms):.3f}")
