# -*- coding: utf-8 -*-
"""Balayage complet (vectorise par groupe KV) : distribution de l'erreur de sortie et
frequence des echecs de la passe 1 (Jensen gap du resume par cle moyenne)."""
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


def sweep(tag, doc=0, Lb=64, W=512, nq=8, m=8, bits=(2, 4, 8)):
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
                Sb = np.einsum("bld,gd->gbl", Kb, qg) / np.sqrt(D)
                mx = Sb.max(2, keepdims=True)
                lse = np.log(np.exp(Sb - mx).sum(2)) + mx[:, :, 0]
                mu = Kb.mean(1)
                sc1 = (qg @ mu.T) / np.sqrt(D)
                to = np.argsort(-lse, axis=1)[:, :m]
                t1 = np.argsort(-sc1, axis=1)[:, :m]
                ko = np.tile(loc, (len(qg), 1)); k1 = ko.copy()
                for gi in range(len(qg)):
                    for b in to[gi]: ko[gi, int(b) * Lb:(int(b) + 1) * Lb] = True
                    for b in t1[gi]: k1[gi, int(b) * Lb:(int(b) + 1) * Lb] = True
                mo = (P_ * ko).sum(1); m1 = (P_ * k1).sum(1); ml = P_[:, loc].sum(1)
                eo = np.linalg.norm(sma(np.where(ko, S, NEG)) @ Kc - O, axis=1) / nO
                for gi in range(len(qg)):
                    r = {"tag": tag, "layer": int(layer), "head": kv * grp + gi, "W": W, "p": int(p),
                         "masse_locale": float(ml[gi]), "masse_oracle": float(mo[gi]),
                         "masse_passe1": float(m1[gi]), "err_sel_oracle": float(eo[gi])}
                    rows.append(r)
                for b in bits:
                    Kq = quant_b(Kb, b).reshape(nbc * Lb, D)
                    Kfull = np.vstack([Kq, Kc[nbc * Lb:]])
                    sq = (qg @ Kfull.T) / np.sqrt(D)
                    eq = np.linalg.norm(sma(sq) @ Kc - O, axis=1) / nO
                    ea = np.linalg.norm(sma(np.where(k1, sq, NEG)) @ Kc - O, axis=1) / nO
                    for gi in range(len(qg)):
                        base = rows[-len(qg) + gi]
                        base[f"quant_{b}"] = float(eq[gi])
                        base[f"asp_{b}"] = float(ea[gi])
    return rows


if __name__ == "__main__":
    allrows = []
    for tag in ("smol8k", "qwen8k"):
        for W in (512, 0):
            r = sweep(tag, W=W)
            allrows += r
            print(f"{tag} W={W}: {len(r)} lignes", flush=True)
    (OUT / "sweep_tetes.json").write_text(json.dumps(allrows), encoding="utf-8")
    print("\n=== distribution par tete (moyenne sur 8 requetes) ===")
    for tag in ("smol8k", "qwen8k"):
        for W in (512, 0):
            sel = [x for x in allrows if x["tag"] == tag and x["W"] == W]
            heads = {}
            for x in sel:
                heads.setdefault((x["layer"], x["head"]), []).append(x)
            agg = {k: {f: float(np.mean([y[f] for y in v])) for f in
                       ("masse_locale", "masse_oracle", "masse_passe1", "err_sel_oracle",
                        "quant_4", "asp_4", "asp_8")} for k, v in heads.items()}
            n = len(agg)
            qf = lambda f, pp: float(np.percentile([v[f] for v in agg.values()], pp))
            print(f"\n{tag} W={W} : {n} tetes")
            for f in ("masse_locale", "masse_oracle", "masse_passe1", "err_sel_oracle",
                      "quant_4", "asp_4", "asp_8"):
                print(f"   {f:16s} med={qf(f,50):8.4f} p90={qf(f,90):8.4f} max={qf(f,100):8.4f}")
            jens = [k for k, v in agg.items() if v["masse_oracle"] > 0.9 and v["masse_passe1"] < 0.5]
            cata = [k for k, v in agg.items() if v["asp_4"] > 0.5]
            print(f"   echecs passe 1 (oracle>0,9 ; passe1<0,5) : {len(jens)}/{n} = {100*len(jens)/n:.1f} %")
            print(f"   catastrophiques (asp_4 > 0,5)              : {len(cata)}/{n} = {100*len(cata)/n:.1f} %")
            if jens: print("   tetes :", sorted(jens)[:12])
