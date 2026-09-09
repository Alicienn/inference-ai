"""
CONCEPT 3 -- Coreset CALIBRE, et test de bout en bout.

Origine. Le diagnostic diag_cert.py etablit deux faits :
  - l'ecart Delta = l_b - lhat_b entre la vraie log-masse et l'estimation du
    coreset est PREVISIBLE (R^2 = 0.72 avec RoPE, 0.81 sans) a partir de deux
    scalaires deja stockes par bloc : le rayon max rho et la plus grande valeur
    propre lambda de la dispersion intra-cluster ;
  - la correction theorique fixe (cs+var, cs+rad) DEGRADE le classement, car ses
    coefficients sont faux. Il faut donc les AJUSTER sur donnees.

Methode. On ajuste par moindres carres, sur un ensemble d'apprentissage disjoint
(couches distinctes), la regression
    Delta ~ a0 + a1 (s||q|| rho) + a2 (s||q||)^2 lambda_max + a3 (s||q||)^2 lambda_moy
puis on corrige le score : ltilde_b = lhat_b + Delta_chapeau_b.
Cout : 2 scalaires par bloc (deja presents pour le certificat) + 4 coefficients
GLOBAUX par couche. C'est-a-dire essentiellement gratuit.

Test de bout en bout. On mesure enfin ce qui compte vraiment : l'erreur sur la
SORTIE d'attention lorsqu'on se restreint aux k blocs selectionnes, et non plus
seulement la qualite du classement.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load
from certified import summary_cert

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"
FEATS = 4


def block_feats(su, qn, s, Lb):
    return np.stack([s * qn * su["rho"].max(),
                     (s * qn) ** 2 * su["lam"].max(),
                     (s * qn) ** 2 * float(su["lam"].mean()),
                     np.full(len(qn), np.log(Lb))], 1)


def gather(tag, Lb, nq, local, use_rope, r, seed, layers=None):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")
    items = []
    for (doc, layer), p in sorted(packs.items()):
        if layers is not None and layer not in layers:
            continue
        K_all, Q_all, V_all = p[kk], p[qq], p[kk]      # valeurs ~ cles (proxy)
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 16:
                continue
            cand = np.arange(1, hi)
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            qn = np.linalg.norm(Q, axis=1)
            lh, tm, F = [], [], []
            for b in cand:
                Kb = K[b * Lb:(b + 1) * Lb]
                su, _ = summary_cert(Kb, r)
                A = s * (Q @ su["mu"].T) + np.log(su["n"])[None, :]
                lh.append(S._lse(A))
                tm.append(S.true_logmass(Kb, Q, s))
                F.append(block_feats(su, qn, s, Lb))
            items.append(dict(K=K, Q=Q, cand=cand, s=s, Lb=Lb,
                              lh=np.stack(lh, 1), tm=np.stack(tm, 1),
                              F=np.stack(F, 1)))
    return items


def fit(items):
    X = np.concatenate([it["F"].reshape(-1, FEATS) for it in items])
    y = np.concatenate([(it["tm"] - it["lh"]).ravel() for it in items])
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def attn_out(K, Q, s, idx):
    """Sortie d'attention restreinte aux indices de cles `idx` (valeurs = cles)."""
    out = np.zeros((len(Q), K.shape[1]), dtype=np.float64)
    for i in range(len(Q)):
        Ks = K[idx[i]]
        z = s * (Q[i] @ Ks.T)
        w = np.exp(z - z.max())
        w /= w.sum()
        out[i] = w @ Ks
    return out


def evaluate(items, coef, topk=(4, 8, 16)):
    res = {}
    for k in topk:
        agg = {m: dict(mrec=[], srec=[], oerr=[]) for m in ("coreset", "calibre", "oracle")}
        for it in items:
            lh, tm, F = it["lh"], it["tm"], it["F"]
            A = np.concatenate([F.reshape(-1, FEATS),
                                np.ones((F.shape[0] * F.shape[1], 1))], 1)
            corr = (A @ coef).reshape(F.shape[0], F.shape[1])
            mass = np.exp(tm - tm.max(1, keepdims=True))
            order = np.argsort(-tm, 1)
            ora = np.take_along_axis(mass, order[:, :k], 1).sum(1)
            K, Q, cand, s, Lb = it["K"], it["Q"], it["cand"], it["s"], it["Lb"]
            # sortie d'attention de reference : sur TOUS les blocs candidats
            allidx = np.concatenate([np.arange(b * Lb, (b + 1) * Lb) for b in cand])
            ref = attn_out(K, Q, s, [allidx] * len(Q))
            for name, sc in (("coreset", lh), ("calibre", lh + corr), ("oracle", tm)):
                om = np.argsort(-sc, 1)[:, :k]
                got = np.take_along_axis(mass, om, 1).sum(1)
                agg[name]["mrec"].append(float(np.mean(got / np.maximum(ora, 1e-30))))
                agg[name]["srec"].append(float(np.mean([
                    len(set(om[i]) & set(order[i, :k])) / k for i in range(len(Q))])))
                idx = [np.concatenate([np.arange(cand[b] * Lb, (cand[b] + 1) * Lb)
                                       for b in om[i]]) for i in range(len(Q))]
                o = attn_out(K, Q, s, idx)
                agg[name]["oerr"].append(float(np.mean(
                    np.linalg.norm(o - ref, axis=1) / np.maximum(
                        np.linalg.norm(ref, axis=1), 1e-9))))
        res[k] = {m: {kk_: float(np.mean(v)) for kk_, v in d.items()}
                  for m, d in agg.items()}
    return res


if __name__ == "__main__":
    meta, packs = load("qwen8k")
    layers = sorted({l for _, l in packs})
    tr, te = layers[::2], layers[1::2]
    print(f"couches d'apprentissage {tr}   couches de test {te}\n")
    out = {}
    for rope in (True, False):
        lbl = "avec RoPE" if rope else "sans RoPE"
        itr = gather("qwen8k", 64, 32, 8, rope, 4, 0, layers=tr)
        ite = gather("qwen8k", 64, 32, 8, rope, 4, 1, layers=te)
        coef = fit(itr)
        res = evaluate(ite, coef)
        print("=" * 96)
        print(f"CONCEPT 3 -- coreset CALIBRE (coefficients appris sur couches disjointes)")
        print(f"{lbl}   coefficients = {np.round(coef,4)}")
        print("=" * 96)
        print(f"{'k':>4s} {'methode':>10s} {'rappel masse':>14s} {'rappel ens.':>13s}"
              f" {'erreur sortie':>15s}")
        print("-" * 96)
        for k, d in res.items():
            for m in ("coreset", "calibre", "oracle"):
                print(f"{k:4d} {m:>10s} {100*d[m]['mrec']:13.2f}% "
                      f"{100*d[m]['srec']:12.2f}% {100*d[m]['oerr']:14.3f}%")
            print()
        out[lbl] = dict(coef=[float(c) for c in coef], res=res)
    (RES / "calibrated.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'calibrated.json'}")
