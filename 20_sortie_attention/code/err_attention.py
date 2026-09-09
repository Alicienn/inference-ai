# -*- coding: utf-8 -*-
"""Erreur de SORTIE d'attention induite par la quantification des cles de score.

Decomposition : (a) selection seule (scores exacts, top-m blocs) ; (b) quantification seule
(attention pleine, scores quantifies) ; (c) pipeline ASP (passe 1 = coreset r=1 = cle
moyenne du bloc, passe 2 quantifiee). Valeurs = cles exactes (proxy V=K).
"""
import numpy as np, json, pathlib
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats")
OUT.mkdir(parents=True, exist_ok=True)


def quant(Kb, bits):
    mn = Kb.min(axis=1, keepdims=True); mx = Kb.max(axis=1, keepdims=True)
    step = np.maximum(mx - mn, 1e-8) / (2 ** bits - 1)
    return np.round((Kb - mn) / step) * step + mn


def sm(S):
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=1, keepdims=True)


def blk_lse(Kb, Qq, scale):
    """score de bloc = log-sum-exp des scores des Lb cles : (nb, nq)."""
    Sb = np.einsum("bld,qd->bql", Kb, Qq) * scale          # (nb, nq, Lb)
    mx = Sb.max(axis=2, keepdims=True)
    return (np.log(np.exp(Sb - mx).sum(axis=2)) + mx[:, :, 0])


def run(tag, doc, layer, head, kvhead, Lb=64, nq=32, m=8, bits=(2, 4, 6, 8)):
    d = np.load(P / f"qk_{tag}.npz", mmap_mode="r")
    K = np.asarray(d[f"{doc}_{layer}_k_rope"])[:, kvhead, :].astype(np.float64)
    Q = np.asarray(d[f"{doc}_{layer}_q_rope"])[:, head, :].astype(np.float64)
    T, D = K.shape
    Qq = Q[T - nq:]
    nb = T // Lb
    Kb = K[:nb * Lb].reshape(nb, Lb, D)
    scale = 1.0 / np.sqrt(D)
    S = (Qq @ K.T) * scale
    Pt = sm(S)
    o = Pt @ K                                              # reference exacte
    res = {"tag": tag, "doc": doc, "layer": layer, "head": head, "T": T, "D": D,
           "nq": nq, "Lb": Lb, "m": m, "bits": list(bits)}

    def mask_from(top):
        keep = np.zeros((nq, T), dtype=bool)
        for i in range(nq):
            for b in top[:, i]:
                b = int(b)
                keep[i, b * Lb:(b + 1) * Lb] = True
        return keep

    # (a) selection seule, oracle
    top_o = np.argsort(-blk_lse(Kb, Qq, scale), axis=0)[:m]
    keep_o = mask_from(top_o)
    om = sm(np.where(keep_o, S, -np.inf)) @ K
    res["err_sel_oracle"] = float(np.mean(np.linalg.norm(om - o, axis=1) / np.linalg.norm(o, axis=1)))
    res["masse_sel_oracle"] = float(np.mean((Pt * keep_o).sum(1)))

    # (b) quantification seule, attention pleine
    res["err_quant"] = {}
    for k in bits:
        Kq = quant(Kb, k).reshape(nb * Lb, D)
        oq = sm((Qq @ Kq.T) * scale) @ K
        res["err_quant"][str(k)] = float(np.mean(np.linalg.norm(oq - o, axis=1) / np.linalg.norm(o, axis=1)))

    # (c) pipeline ASP : passe 1 = coreset r=1, passe 2 quantifiee
    mu = Kb.mean(axis=1)
    sc1 = (Qq @ mu.T) * scale
    keep_1 = mask_from(np.argsort(-sc1, axis=0)[:m])
    res["masse_passe1"] = float(np.mean((Pt * keep_1).sum(1)))
    res["err_asp"] = {}
    for k in bits:
        Kq = quant(Kb, k).reshape(nb * Lb, D)
        oa = sm(np.where(keep_1, (Qq @ Kq.T) * scale, -np.inf)) @ K
        res["err_asp"][str(k)] = float(np.mean(np.linalg.norm(oa - o, axis=1) / np.linalg.norm(o, axis=1)))
    return res


if __name__ == "__main__":
    d = np.load(P / "qk_smol8k.npz", mmap_mode="r")
    meta = json.loads(str(d["meta"]))
    H, KVH = meta["H"], meta["KVH"]
    print(f"smol8k : H={H} KVH={KVH} D={meta['D']} layers={meta['layers']}")
    out = []
    for layer in meta["layers"][:4]:
        for head in (0, H // 2):
            out.append(run("smol8k", 0, layer, head, head // (H // KVH)))
    for r in out:
        print(f"  L{r['layer']:2d} h{r['head']} | sel_oracle={r['err_sel_oracle']:.4f} (masse {r['masse_sel_oracle']:.4f}) "
              f"| masse passe1={r['masse_passe1']:.4f} | quant " +
              " ".join(f"{k}b:{v:.4f}" for k, v in r["err_quant"].items()) + " | ASP " +
              " ".join(f"{k}b:{v:.4f}" for k, v in r["err_asp"].items()))
    (OUT / "err_attention_smol.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", OUT / "err_attention_smol.json")
