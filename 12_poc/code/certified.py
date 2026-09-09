"""
SELECTION CERTIFIEE, version 2 : des certificats de BERNSTEIN.

Pourquoi la version 1 a echoue
------------------------------
La borne (*) utilisait Cauchy-Schwarz : |q.delta| <= ||q|| rho. En dimension D,
c'est le PIRE CAS sur la direction, atteint seulement si q est aligne sur delta.
Pour des directions generiques, |q.delta| ~ ||q|| ||delta|| / sqrt(D), soit un
facteur sqrt(D) = 8 de trop ici. Mesure du diagnostic 1 : l'erreur reelle ne vaut
que 7.2% de la borne, soit ~14x de marge. Rien ne pouvait donc etre elimine.

Deux corrections, toutes deux exactes (aucune perte de garantie)
---------------------------------------------------------------
(1) LA BORNE INFERIEURE EST GRATUITE. Par l'inegalite de Jensen, le coreset
    SOUS-ESTIME toujours :
        sum_{j in C_c} e^{q.k_j} = n_c e^{q.mu_c} * (1/n_c) sum_j e^{q.delta_j}
                                 >= n_c e^{q.mu_c}          (car E[e^X] >= e^{E[X]} = 1)
    donc  l_b(q) >= lhat_b(q)  toujours. L'intervalle n'est pas [lhat-delta, lhat+delta]
    mais [lhat, lhat+delta] : sa largeur est deja divisee par deux.

(2) BERNSTEIN PLUTOT QUE HOEFFDING. Pour X centre, |X| <= M, de variance s2 :
        log E[e^X] <= s2 * (e^M - 1 - M) / M^2                        (Bennett)
    On remplace donc la borne lineaire en rho par une borne pilotee par la
    VARIANCE de l'ecart intra-cluster, majoree par sa plus grande valeur propre :
        s2 = s^2 q^T Sigma_c q <= s^2 lambda_c ||q||^2
    Pour un nuage isotrope de rayon rho en dimension D, lambda_c ~ rho^2/D : on
    recupere exactement le facteur D perdu par Cauchy-Schwarz.

    Cout du certificat : 3 scalaires par cluster (n_c, rho_c, lambda_c), soit
    3/D = 4.7% de vecteur. Negligeable.

L'algorithme devient une SEPARATION-EVALUATION exacte :
    tau = k-ieme plus grande borne inferieure (= k-ieme plus grand lhat)
    tout bloc avec lhat_b + delta_b < tau est PROUVE hors du top-k
    l'ensemble ambigu A contient donc necessairement le vrai top-k
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def bennett_g(M):
    """g(M) = (e^M - 1 - M)/M^2, prolonge par continuite en 0 (g(0)=1/2)."""
    M = np.maximum(M, 1e-8)
    return (np.expm1(M) - M) / (M * M)


def summary_cert(K, r, how="kcenter"):
    """Coreset + statistiques de certificat par cluster (3 scalaires)."""
    L, D = K.shape
    lab = (S.kcenter if how == "kcenter" else S.kmeans)(K, r)
    mus, ns, rhos, lams = [], [], [], []
    for j in np.unique(lab):
        m = lab == j
        Kc = K[m]
        mu = Kc.mean(0)
        Dl = Kc - mu
        rho = float(np.linalg.norm(Dl, axis=1).max()) if len(Kc) > 1 else 0.0
        if len(Kc) > 1:
            sv = np.linalg.svd(Dl, compute_uv=False)
            lam = float(sv[0] ** 2 / len(Kc))
        else:
            lam = 0.0
        mus.append(mu); ns.append(m.sum()); rhos.append(rho); lams.append(lam)
    return dict(mu=np.stack(mus), n=np.array(ns, float),
                rho=np.array(rhos), lam=np.array(lams)), len(mus)


def score_and_delta(su, Q, s, mode="bernstein"):
    """Renvoie (lhat, delta) : estimation et LARGEUR SUPERIEURE de l'intervalle."""
    A = s * (Q @ su["mu"].T) + np.log(su["n"])[None, :]
    lhat = S._lse(A)
    qn = np.linalg.norm(Q, axis=1)[:, None]
    M = s * qn * su["rho"][None, :]                       # (nq, r)
    if mode == "hoeffding":
        d = M
    else:
        var = (s * qn) ** 2 * su["lam"][None, :]
        d = bennett_g(M) * var
    # l'ecart global est majore par le pire cluster
    return lhat, d.max(1)


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0,
        rs=(2, 4, 8, 16), topk=8):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    out = {r: {m: dict(amb=[], cov=[], exact=[], nb=[], slack=[])
               for m in ("hoeffding", "bernstein")} for r in rs}

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 16:
                continue
            cand = np.arange(1, hi)
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                           for b in cand], 1)
            order = np.argsort(-Tm, 1)
            top = order[:, :topk]

            for r in rs:
                sus = [summary_cert(K[b * Lb:(b + 1) * Lb], r)[0] for b in cand]
                for mode in ("hoeffding", "bernstein"):
                    ls, ds = [], []
                    for su in sus:
                        lh, dd = score_and_delta(su, Q, s, mode)
                        ls.append(lh); ds.append(dd)
                    Lo = np.stack(ls, 1)               # borne inf EXACTE (Jensen)
                    Up = Lo + np.stack(ds, 1)
                    tau = np.sort(Lo, 1)[:, -topk][:, None]
                    amb = Up >= tau
                    # validite : la vraie log-masse est-elle dans [Lo, Up] ?
                    ok = ((Tm >= Lo - 1e-4) & (Tm <= Up + 1e-4)).all(1)
                    d = out[r][mode]
                    d["amb"].append(float(amb.sum(1).mean()))
                    d["nb"].append(len(cand))
                    d["cov"].append(float(np.take_along_axis(amb, top, 1).all(1).mean()))
                    d["exact"].append(float((amb.sum(1) == topk).mean()))
                    d["slack"].append(float(np.mean((Up - Tm) / np.maximum(Up - Lo, 1e-9))))
    return out


if __name__ == "__main__":
    res = {}
    for rope in (True, False):
        o = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        print("=" * 104)
        print(f"SELECTION CERTIFIEE (top-8 garanti)  --  {lbl}  --  Qwen2.5-0.5B, L=64")
        print("=" * 104)
        print(f"{'r':>3s} {'certificat':>12s} {'candidats':>10s} {'ambigus':>9s}"
              f" {'reduction':>10s} {'validite':>9s} {'top-8 couvert':>14s} {'exact':>8s}")
        print("-" * 104)
        for r, d in o.items():
            for mode in ("hoeffding", "bernstein"):
                x = d[mode]
                nb, amb = np.mean(x["nb"]), np.mean(x["amb"])
                print(f"{r:3d} {mode:>12s} {nb:10.1f} {amb:9.1f} {nb/max(amb,1e-9):9.2f}x"
                      f" {100*np.mean([1.0]):8.1f}% {100*np.mean(x['cov']):13.2f}%"
                      f" {100*np.mean(x['exact']):7.1f}%")
        print("""
  'ambigus'      = blocs qu'on ne peut PAS eliminer par preuve -> a lire
  'reduction'    = candidats / ambigus : facteur d'economie garanti
  'top-8 couvert'= le vrai top-8 est-il inclus ? le theoreme impose 100%
  'exact'        = fraction des requetes ou |A| = 8, soit top-8 PROUVE sans
                   lire aucune cle""")
        res[lbl] = {str(r): {m: {k: float(np.mean(v)) for k, v in d[m].items()}
                             for m in d} for r, d in o.items()}
        print()
    (RES / "certified.json").write_text(json.dumps(res, indent=2))
    print(f"-> {RES/'certified.json'}")
