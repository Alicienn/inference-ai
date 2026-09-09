# -*- coding: utf-8 -*-
"""Reconciliation : les erreurs catastrophiques du resume par cle moyenne viennent-elles
du BLOC DU SINK (position 0) manque ? Methodes, passe 2 = 4 bits, W=512, m=8, causal :
  1. mean_r1            : top-m par score de la cle moyenne (baseline)
  2. mean_r1 + sink     : idem mais bloc 0 force dans le jeu garde (m-1 autres)
  3. s1_off0            : top-m par score de la 1re cle de chaque bloc
  4. s1_off16 + sink    : offset 16, bloc 0 force
  5. oracle             : top-m par LSE exact
"""
import numpy as np, json, pathlib
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats")
NEG = -1e30
METHODS = ("mean_r1", "mean_r1_sink", "s1_off0", "s1_off16_sink", "oracle")


def sma(S):
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(1, keepdims=True)


def quant_b(Kb, bits):
    mn = Kb.min(axis=1, keepdims=True); mx = Kb.max(axis=1, keepdims=True)
    step = np.maximum(mx - mn, 1e-8) / (2 ** bits - 1)
    return np.round((Kb - mn) / step) * step + mn


def topm_with_sink(sc, m, force_sink):
    """Retourne les indices de blocs selectionnes ; si force_sink, le bloc 0 est toujours garde."""
    if not force_sink:
        return np.argsort(-sc, axis=1)[:, :m]
    order = np.argsort(-sc, axis=1)
    out = np.zeros((sc.shape[0], m), dtype=int)
    out[:, 0] = 0
    for gi in range(sc.shape[0]):
        rest = [b for b in order[gi] if b != 0][:m - 1]
        out[gi, 1:] = rest
    return out


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
                mx = Sb.max(2, keepdims=True)
                lse = np.log(np.exp(Sb - mx).sum(2)) + mx[:, :, 0]
                mu = Kb.mean(1)
                sc_mean = (qg @ mu.T) / np.sqrt(D)
                sc_off0 = Sb[:, :, 0]
                sc_off16 = Sb[:, :, 16]
                base = len(rows)
                for gi in range(len(qg)):
                    rows.append({"tag": tag, "layer": int(layer), "head": kv * grp + gi,
                                 "masse_sink": float(P_[gi, :Lb].sum())})
                for name, sc, fs in (("mean_r1", sc_mean, False), ("mean_r1_sink", sc_mean, True),
                                     ("s1_off0", sc_off0, False), ("s1_off16_sink", sc_off16, True),
                                     ("oracle", lse, False)):
                    tm = topm_with_sink(sc, m, fs)
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
    (OUT / "reconciliation_sink.json").write_text(json.dumps(allrows), encoding="utf-8")
    for tag in ("smol8k", "qwen8k"):
        sel = [x for x in allrows if x["tag"] == tag]
        heads = {}
        for x in sel:
            heads.setdefault((x["layer"], x["head"]), []).append(x)
        sk = np.mean([np.mean([y["masse_sink"] for y in v]) for v in heads.values()])
        print(f"\n{tag} : {len(heads)} tetes | masse du bloc du sink = {sk:.4f}")
        for name in METHODS:
            e = np.array([np.mean([y[f"err_{name}"] for y in v]) for v in heads.values()])
            ms = np.array([np.mean([y[f"mass_{name}"] for y in v]) for v in heads.values()])
            print(f"   {name:15s} err med={np.median(e):.4f} p90={np.percentile(e,90):.4f} "
                  f"max={e.max():.4f} | catastrophes={100*(e>0.5).mean():5.1f} % | masse med={np.median(ms):.3f}")
