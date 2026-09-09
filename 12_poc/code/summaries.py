"""
Resumes de blocs pour la selection d'attention creuse, et leurs scores.

CADRE THEORIQUE
---------------
La masse d'attention d'un bloc b vaut  m_b(q) = sum_{r in b} exp(s * q.k_r).
En posant l_b(q) = log m_b(q), on a l'identite variationnelle de Gibbs :

    l_b(q) = max_{p dans le simplexe} [ sum_r p_r (s q.k_r) + H(p) ]

l_b est donc un "max adouci" : il interpole entre  log L + moyenne  (q -> 0)
et  max_r s q.k_r  (||q|| -> infini). Deux familles d'approximation en decoulent.

(A) APPROXIMATION PAR MOMENTS  [COBS, arXiv 2607.09052]
    Developpement de Taylor de la fonction generatrice des cumulants en q = 0 :
        l_b ~ log L + q.kbar + 1/2 q^T Sigma q + (1/6) kappa3[q,q,q] + ...
    Valide pour ||q||*dispersion petit. Pour une distribution ATOMIQUE avec une
    valeur aberrante, la serie est ASYMPTOTIQUE : la tronquer plus loin peut
    empirer -- c'est exactement le resultat negatif de COBS sur le 3e cumulant
    (MK3 0.34 -> 0.01). De plus le terme quadratique est AVEUGLE AU SIGNE :
    q^T Sigma q ne distingue pas +u de -u.

(B) APPROXIMATION PAR SUPPORT  [notre proposition]
    On approxime la GEOMETRIE du nuage de cles par un coreset de r centroides.
    Si les cles sont partitionnees en C_1..C_r de centroides mu_c, effectifs n_c
    et rayons rho_c = max_{k dans C_c} ||k - mu_c||, alors pour tout q :

        | l_b(q) - log sum_c n_c exp(s q.mu_c) |  <=  s ||q|| max_c rho_c      (*)

    PREUVE. Pour chaque c et chaque k de C_c, Cauchy-Schwarz donne
    |q.(k - mu_c)| <= ||q|| rho_c, d'ou
        n_c e^{s q.mu_c - s||q||rho_c}  <=  sum_{k in C_c} e^{s q.k}
                                        <=  n_c e^{s q.mu_c + s||q||rho_c}.
    En sommant sur c puis en prenant le logarithme, les facteurs communs
    e^{+-s||q||rho} se factorisent et bornent l'ecart. CQFD

    (*) est une borne UNIFORME sur la direction de q, la ou le developpement en
    cumulants n'offre qu'un controle local autour de 0. Elle est de plus SIGNEE
    (exp(q.mu_c) distingue +mu de -mu) et EXACTE sur une aiguille isolee, qui
    forme un singleton de rayon nul. Le bon objectif de partition est donc le
    k-CENTRE (minimiser le rayon maximal), et non le k-moyennes.

(C) FAMILLE UNIFIEE : melange gaussien par bloc
        l_b ~ log sum_c n_c exp(s q.mu_c + (s^2/2) q^T Sigma_c q)
    COBS est le cas r = 1. Le coreset est le cas Sigma_c = 0. La question de
    recherche devient : a budget d'octets fixe, comment repartir le budget entre
    NOMBRE DE COMPOSANTES et RANG DE COVARIANCE ?
"""
import numpy as np


# --------------------------------------------------------------------------
# Partitionnement
# --------------------------------------------------------------------------
def kcenter(K, r, seed=0):
    """Gonzalez : 2-approximation du k-centre (minimise le rayon maximal)."""
    L = K.shape[0]
    if r >= L:
        return np.arange(L)
    c = [int(np.argmax(np.linalg.norm(K - K.mean(0), axis=1)))]
    d = np.linalg.norm(K - K[c[0]], axis=1)
    for _ in range(r - 1):
        i = int(np.argmax(d))
        c.append(i)
        d = np.minimum(d, np.linalg.norm(K - K[i], axis=1))
    C = np.stack([K[i] for i in c])
    return np.argmin(((K[:, None, :] - C[None]) ** 2).sum(-1), 1)


def kmeans(K, r, iters=15, seed=0):
    """Lloyd avec initialisation k-means++ (rayon quadratique moyen)."""
    L = K.shape[0]
    if r >= L:
        return np.arange(L)
    rng = np.random.default_rng(seed)
    C = [K[rng.integers(L)]]
    for _ in range(r - 1):
        d = np.min(((K[:, None, :] - np.stack(C)[None]) ** 2).sum(-1), 1)
        tot = d.sum()
        p = d / tot if tot > 0 else np.full(L, 1.0 / L)
        C.append(K[rng.choice(L, p=p)])
    C = np.stack(C)
    lab = np.zeros(L, dtype=int)
    for _ in range(iters):
        lab = np.argmin(((K[:, None, :] - C[None]) ** 2).sum(-1), 1)
        for j in range(r):
            m = lab == j
            if m.any():
                C[j] = K[m].mean(0)
    return lab


def max_radius(K, lab):
    """Rayon maximal de la partition : la quantite qui controle la borne (*)."""
    rho = 0.0
    for j in np.unique(lab):
        m = lab == j
        mu = K[m].mean(0)
        rho = max(rho, float(np.linalg.norm(K[m] - mu, axis=1).max()))
    return rho


# --------------------------------------------------------------------------
# Construction des resumes.
# Chaque fonction renvoie (resume, cout) ou le cout est exprime en nombre de
# vecteurs de dimension D ; un scalaire compte pour 1/D.
# --------------------------------------------------------------------------
def s_mean(K):
    return dict(kind="mean", mu=K.mean(0)), 1.0


def s_maxpool(K):
    return dict(kind="mean", mu=K.max(0)), 1.0


def s_quest(K):
    return dict(kind="quest", lo=K.min(0), hi=K.max(0)), 2.0


def s_cobs(K, r):
    """Moyenne + r directions propres de la covariance intra-bloc (COBS)."""
    L, D = K.shape
    mu = K.mean(0)
    X = K - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(r, Vt.shape[0])
    lam = (S[:r] ** 2) / L
    return dict(kind="cobs", mu=mu, U=Vt[:r], lam=lam), 1.0 + r + r / D


def s_coreset(K, r, how="kcenter"):
    """r centroides + effectifs ; scoring par log-sum-exp exact du melange."""
    L, D = K.shape
    lab = (kcenter if how == "kcenter" else kmeans)(K, r)
    mus, ns = [], []
    for j in np.unique(lab):
        m = lab == j
        mus.append(K[m].mean(0))
        ns.append(m.sum())
    nk = len(mus)
    return dict(kind="coreset", mu=np.stack(mus), n=np.array(ns, float),
                rho=max_radius(K, lab)), nk * (1.0 + 1.0 / D)


def s_gmm(K, r, rank=1, how="kcenter"):
    """Melange : r composantes, chacune moyenne + covariance de rang `rank`."""
    L, D = K.shape
    lab = (kcenter if how == "kcenter" else kmeans)(K, r)
    mus, ns, Us, lams = [], [], [], []
    for j in np.unique(lab):
        m = lab == j
        Kc = K[m]
        mu = Kc.mean(0)
        X = Kc - mu
        if Kc.shape[0] > 1 and rank > 0:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            rr = min(rank, Vt.shape[0])
            u = Vt[:rr]
            lm = (S[:rr] ** 2) / Kc.shape[0]
            if rr < rank:
                u = np.vstack([u, np.zeros((rank - rr, D))])
                lm = np.concatenate([lm, np.zeros(rank - rr)])
        else:
            u = np.zeros((max(rank, 1), D))
            lm = np.zeros(max(rank, 1))
        mus.append(mu); ns.append(m.sum()); Us.append(u); lams.append(lm)
    nk = len(mus)
    return dict(kind="gmm", mu=np.stack(mus), n=np.array(ns, float),
                U=np.stack(Us), lam=np.stack(lams)), \
        nk * (1.0 + rank + (1.0 + rank) / D)




def s_coreset_var(K, r, how="kcenter"):
    """
    Coreset + correction de variance ISOTROPE (1 scalaire de plus par cluster).

    Le coreset nu sous-estime systematiquement, car il ignore la dispersion
    intra-cluster :
        sum_{k in C_c} e^{s q.k} = n_c e^{s q.mu_c} * (1/n_c) sum e^{s q.delta}
                                 ~ n_c e^{s q.mu_c} (1 + (s^2/2) q^T Sigma_c q).
    Ce biais depend de la dispersion du cluster, donc il n'est PAS uniforme entre
    blocs : il sous-classe precisement les blocs a forte dispersion, qui sont
    ceux qui portent une aiguille. On corrige avec l'approximation isotrope
    Sigma_c ~ (tr Sigma_c / D) I, soit UN scalaire par cluster :
        (s^2/2) ||q||^2 tr(Sigma_c) / D
    """
    L, D = K.shape
    lab = (kcenter if how == "kcenter" else kmeans)(K, r)
    mus, ns, tr = [], [], []
    for j in np.unique(lab):
        m = lab == j
        Kc = K[m]; mu = Kc.mean(0)
        mus.append(mu); ns.append(m.sum())
        tr.append(float(((Kc - mu) ** 2).sum(1).mean()))
    nk = len(mus)
    return dict(kind="coreset_var", mu=np.stack(mus), n=np.array(ns, float),
                tr=np.array(tr), D=D), nk * (1.0 + 2.0 / D)


def s_coreset_rad(K, r, alpha=1.0, how="kcenter"):
    """
    Coreset + borne de rayon (1 scalaire par cluster). Correction OPTIMISTE
    issue directement de la borne (*) : e^{s q.mu_c} -> e^{s q.mu_c + alpha s ||q|| rho_c}.
    A alpha = 1 c'est la borne superieure exacte du theoreme ; alpha < 1 l'assouplit.
    C'est la forme principielle de l'optimisme grossier de Quest.
    """
    L, D = K.shape
    lab = (kcenter if how == "kcenter" else kmeans)(K, r)
    mus, ns, rad = [], [], []
    for j in np.unique(lab):
        m = lab == j
        Kc = K[m]; mu = Kc.mean(0)
        mus.append(mu); ns.append(m.sum())
        rad.append(float(np.linalg.norm(Kc - mu, axis=1).max()))
    nk = len(mus)
    return dict(kind="coreset_rad", mu=np.stack(mus), n=np.array(ns, float),
                rho=np.array(rad), alpha=alpha), nk * (1.0 + 2.0 / D)


# --------------------------------------------------------------------------
# Scores.  Q : (nq, D). Renvoie (nq,) le log-masse estime du bloc.
# --------------------------------------------------------------------------
def _lse(A):
    M = A.max(1, keepdims=True)
    return (M + np.log(np.exp(A - M).sum(1, keepdims=True)))[:, 0]


def score(summ, Q, s):
    k = summ["kind"]
    if k == "mean":
        return s * (Q @ summ["mu"])
    if k == "quest":
        return s * (np.maximum(Q, 0) @ summ["hi"] + np.minimum(Q, 0) @ summ["lo"])
    if k == "cobs":
        lin = s * (Q @ summ["mu"])
        proj = s * (Q @ summ["U"].T)
        return lin + 0.5 * (proj ** 2 * summ["lam"]).sum(1)
    if k == "coreset":
        A = s * (Q @ summ["mu"].T) + np.log(summ["n"])[None, :]
        return _lse(A)
    if k == "coreset_var":
        qn2 = (Q ** 2).sum(1)[:, None]
        A = (s * (Q @ summ["mu"].T) + np.log(summ["n"])[None, :]
             + 0.5 * s * s * qn2 * (summ["tr"] / summ["D"])[None, :])
        return _lse(A)
    if k == "coreset_rad":
        qn = np.linalg.norm(Q, axis=1)[:, None]
        A = (s * (Q @ summ["mu"].T) + np.log(summ["n"])[None, :]
             + summ["alpha"] * s * qn * summ["rho"][None, :])
        return _lse(A)
    if k == "gmm":
        lin = s * (Q @ summ["mu"].T)
        proj = s * np.einsum("qd,rkd->qrk", Q, summ["U"])
        quad = 0.5 * (proj ** 2 * summ["lam"][None]).sum(-1)
        return _lse(lin + quad + np.log(summ["n"])[None, :])
    raise ValueError(k)


def true_logmass(K, Q, s):
    """Log-masse exacte : log sum_r exp(s q.k_r)."""
    return _lse(s * (Q @ K.T))
