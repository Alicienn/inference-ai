# -*- coding: utf-8 -*-
"""Erreur de SORTIE d'attention : banc correct.
Masque causal, fenetre locale exacte (W tokens) + m blocs lointains selectionnes.
Mesure : (a) selection oracle, (b) quantification seule, (c) pipeline ASP (passe 1 = cle
moyenne de bloc, passe 2 quantifiee). Valeurs = cles exactes (proxy V=K).
"""
import numpy as np, json, pathlib
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats")


def quant(Kb, bits):
    mn = Kb.min(axis=1, keepdims=True); mx = Kb.max(axis=1, keepdims=True)
    step = np.maximum(mx - mn, 1e-8) / (2 ** bits - 1)
    return np.round((Kb - mn) / step) * step + mn


def sm(S):
    S = S - S.max()
    e = np.exp(S)
    return e / e.sum()


def run(tag, doc, layer, head, kvhead, Lb=64, W=512, nq=16, m=8, bits=(2, 4, 6, 8)):
    d = np.load(P / f"qk_{tag}.npz", mmap_mode="r")
    K = np.asarray(d[f"{doc}_{layer}_k_rope"])[:, kvhead, :].astype(np.float64)
    Q = np.asarray(d[f"{doc}_{layer}_q_rope"])[:, head, :].astype(np.float64)
    T, D = K.shape
    scale = 1.0 / np.sqrt(D)
    Kb_all = K[: (T // Lb) * Lb].reshape(T // Lb, Lb, D)
    acc = {k: [] for k in ("sel_oracle", "sel_oracle_noloc", "quant_full", "asp")}
    acc.update({f"quant_full_{b}": [] for b in bits})
    acc.update({f"asp_{b}": [] for b in bits})
    mass_o, mass_1, mass_loc = [], [], []
    for i in range(nq):
        p = T - nq + i
        q = Q[p]
        Kc = K[:p + 1]
        S = (Kc @ q) * scale
        Pt = sm(S)
        o = Pt @ Kc
        loc = np.zeros(p + 1, dtype=bool)
        loc[max(0, p - W + 1):p + 1] = True
        nbc = max(0, (p - W + 1) // Lb)              # blocs entierement avant la fenetre
        if nbc < m:
            continue
        Kb = Kb_all[:nbc]
        # scores de bloc exacts (log-sum-exp) et passe 1 (cle moyenne)
        Sb = np.einsum("bld,d->bl", Kb, q) * scale
        mx = Sb.max(axis=1, keepdims=True)
        lse = (np.log(np.exp(Sb - mx).sum(axis=1)) + mx[:, 0])
        mu = Kb.mean(axis=1)
        sc1 = (mu @ q) * scale
        top_o = np.argsort(-lse)[:m]
        top_1 = np.argsort(-sc1)[:m]
        keep_o = loc.copy()
        for b in top_o:
            keep_o[int(b) * Lb:(int(b) + 1) * Lb] = True
        keep_1 = loc.copy()
        for b in top_1:
            keep_1[int(b) * Lb:(int(b) + 1) * Lb] = True
        mass_o.append(Pt[keep_o].sum()); mass_1.append(Pt[keep_1].sum())
        mass_loc.append(Pt[loc].sum())
        oo = sm(np.where(keep_o, S, -np.inf)) @ Kc
        acc["sel_oracle"].append(np.linalg.norm(oo - o) / np.linalg.norm(o))
        # sans fenetre locale (top-m blocs seuls, sur tous les blocs causals)
        nb_all = (p + 1) // Lb
        if nb_all >= m:
            Kb2 = Kb_all[:nb_all]
            Sb2 = np.einsum("bld,d->bl", Kb2, q) * scale
            mx2 = Sb2.max(axis=1, keepdims=True)
            lse2 = np.log(np.exp(Sb2 - mx2).sum(axis=1)) + mx2[:, 0]
            keep_n = np.zeros(p + 1, dtype=bool)
            for b in np.argsort(-lse2)[:m]:
                keep_n[int(b) * Lb:(int(b) + 1) * Lb] = True
            acc["sel_oracle_noloc"].append(
                np.linalg.norm(sm(np.where(keep_n, S, -np.inf)) @ Kc - o) / np.linalg.norm(o))
        for b in bits:
            Kq = quant(Kb_all[: (p + 1) // Lb], b).reshape(((p + 1) // Lb) * Lb, D)
            Kqf = np.vstack([Kq, Kc[(p + 1) // Lb * Lb:]])
            acc[f"quant_full_{b}"].append(
                np.linalg.norm(sm((Kqf @ q) * scale) @ Kc - o) / np.linalg.norm(o))
            Sq = np.where(keep_1, (Kqf @ q) * scale, -np.inf)
            acc[f"asp_{b}"].append(np.linalg.norm(sm(Sq) @ Kc - o) / np.linalg.norm(o))
    res = {"tag": tag, "doc": doc, "layer": layer, "head": head, "T": T, "Lb": Lb, "W": W,
           "m": m, "nq": len(acc["sel_oracle"]), "bits": list(bits),
           "masse_locale": float(np.mean(mass_loc)), "masse_sel_oracle": float(np.mean(mass_o)),
           "masse_passe1": float(np.mean(mass_1))}
    for k, v in acc.items():
        if v:
            res[k] = float(np.mean(v))
    return res


if __name__ == "__main__":
    d = np.load(P / "qk_smol8k.npz", mmap_mode="r")
    meta = json.loads(str(d["meta"])); H, KVH = meta["H"], meta["KVH"]
    out = []
    for W in (512, 0):
        for layer in meta["layers"][:4]:
            for head in (0, H // 2):
                r = run("smol8k", 0, layer, head, head // (H // KVH), W=W)
                r["W"] = W
                out.append(r)
                print(f"  W={W:3d} L{r['layer']:2d} h{r['head']} | masse loc={r['masse_locale']:.3f} "
                      f"oracle={r['masse_sel_oracle']:.3f} passe1={r['masse_passe1']:.3f} | "
                      f"err sel={r['sel_oracle']:.4f} (sans loc {r.get('sel_oracle_noloc', float('nan')):.4f}) | "
                      f"quant " + " ".join(f"{b}b:{r['quant_full_'+str(b)]:.4f}" for b in (2, 4, 6, 8)) + " | "
                      f"ASP " + " ".join(f"{b}b:{r['asp_'+str(b)]:.4f}" for b in (2, 4, 6, 8)))
    (OUT / "err_attention_smol_v2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", OUT / "err_attention_smol_v2.json")
