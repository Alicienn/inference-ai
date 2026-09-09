# -*- coding: utf-8 -*-
"""Quel resume de passe 1 repare l'erreur de sortie ? Comparaison a budget d'octets donne :
mean r=1 (260 o), top-norme (260 o), k-centre r=2 (520 o), k-centre r=4 (1040 o), oracle (LSE exact).
Passe 2 = 4 bits pour tous. Fenetre locale W=512, m=8, causal."""
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


def kcenter(Kb, r, iters=3, seed=0):
    nb, Lb, D = Kb.shape
    rng = np.random.default_rng(seed)
    C = np.take_along_axis(Kb, rng.integers(0, Lb, size=(nb, r))[:, :, None], axis=1).copy()
    for _ in range(iters):
        d2 = ((Kb[:, :, None, :] - C[:, None, :, :]) ** 2).sum(-1)
        a = d2.argmin(-1)
        for j in range(r):
            m = (a == j)[:, :, None]
            cnt = np.maximum(m.sum(1), 1)
            C[:, j] = (Kb * m).sum(1) / cnt
    cnt = np.stack([(a == j).sum(1) for j in range(r)], 1).astype(np.float64)
    return C, np.maximum(cnt, 1.0)


def lse_scores(C, cnt, qg, D):
    """score = logsumexp_j(q.c_j + log n_j) -> (Hg, nb)."""
    sc = np.einsum("bjd,gd->gbj", C, qg) / np.sqrt(D) + np.log(cnt)[None, :, :]
    mx = sc.max(2, keepdims=True)
    return np.log(np.exp(sc - mx).sum(2)) + mx[:, :, 0]


def sweep(tag, doc=0, Lb=64, W=512, nq=5, m=8, bits=4):
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
                mu = Kb.mean(1)[:, None, :]
                cnt1 = np.full((nbc, 1), float(Lb))
                top = Kb[np.arange(nbc), np.linalg.norm(Kb, axis=2).argmax(1)][:, None, :]
                C2, n2 = kcenter(Kb, 2)
                C4, n4 = kcenter(Kb, 4)
                methods = {
                    "mean_r1": (mu, cnt1),
                    "topnorm": (top, cnt1),
                    "kcenter_r2": (C2, n2),
                    "kcenter_r4": (C4, n4),
                }
                Sb = np.einsum("bld,gd->gbl", Kb, qg) / np.sqrt(D)
                mx = Sb.max(2, keepdims=True)
                oracle_sc = np.log(np.exp(Sb - mx).sum(2)) + mx[:, :, 0]
                base = len(rows)
                for gi in range(len(qg)):
                    rows.append({"tag": tag, "layer": int(layer), "head": kv * grp + gi, "p": int(p),
                                 "masse_locale": float(P_[gi, loc].sum())})
                for name, val in list(methods.items()) + [("oracle", None)]:
                    sc = oracle_sc if val is None else lse_scores(val[0], val[1], qg, D)
                    tm = np.argsort(-sc, axis=1)[:, :m]
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
    (OUT / "pass1_compare.json").write_text(json.dumps(allrows), encoding="utf-8")
    print("\n=== par tete (moyenne sur 6 requetes), passe 2 = 4 bits, W=512, m=8 ===")
    for tag in ("smol8k", "qwen8k"):
        sel = [x for x in allrows if x["tag"] == tag]
        heads = {}
        for x in sel:
            heads.setdefault((x["layer"], x["head"]), []).append(x)
        agg = {k: {f: float(np.mean([y[f] for y in v])) for f in v[0] if f.startswith(("mass_", "err_"))}
               for k, v in heads.items()}
        n = len(agg)
        print(f"\n{tag} : {n} tetes   (octets/bloc : mean_r1 260, topnorm 260, r2 520, r4 1040)")
        for name in ("mean_r1", "topnorm", "kcenter_r2", "kcenter_r4", "oracle"):
            e = np.array([v[f"err_{name}"] for v in agg.values()])
            ms = np.array([v[f"mass_{name}"] for v in agg.values()])
            print(f"   {name:12s} err med={np.median(e):.4f} p90={np.percentile(e,90):.4f} "
                  f"max={e.max():.4f} | catastrophes={100*(e>0.5).mean():5.1f} % | masse med={np.median(ms):.3f}")
