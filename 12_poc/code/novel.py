"""
DEUX CONCEPTS NOUVEAUX pour la selection de blocs.

================================================================================
CADRE : resumer un bloc, c'est QUANTIFIER UNE MESURE sous la metrique log-sum-exp
================================================================================
Un bloc n'est rien d'autre qu'une mesure empirique sur l'espace des cles :
        mu_b = sum_{j dans b} delta_{k_j}
et sa masse d'attention est la transformee de Laplace de cette mesure :
        m_b(q) = integrale e^{s q.k} d mu_b(k).
Resumer un bloc = remplacer mu_b par une mesure a peu d'atomes nu = sum_c w_c delta_{z_c}
telle que les deux transformees de Laplace coincident sur la loi des requetes.

C'est donc un probleme de QUADRATURE a noyau exponentiel, ou les requetes jouent
le role des fonctions test. Cette lecture unifie tout l'etat de l'art :
  - mean-pool : quadrature a 1 noeud, poids impose, ne matche que l'ordre 1
  - COBS      : matche les ordres 1 et 2 (moyenne + covariance), mais en
                DEPENSANT son budget dans l'espace des MOMENTS
  - coreset   : quadrature a r noeuds, poids = effectifs (imposes)
Or la theorie classique de la quadrature dit qu'avec r noeuds et des POIDS
LIBRES on peut matcher bien plus de moments qu'avec des poids imposes (c'est
l'idee de Gauss : r noeuds bien places + r poids libres = exactitude a l'ordre
2r-1). D'ou le concept 1.

================================================================================
CONCEPT 1 -- EQS : Resume par Quadrature Exponentielle
================================================================================
On garde les r centroides mais on OPTIMISE les poids w_c pour minimiser l'erreur
relative de la transformee de Laplace sur la loi des requetes. Points cles :
  - le probleme est un MOINDRES CARRES LINEAIRE en w (la masse est lineaire en w),
    donc resolu exactement, sans descente de gradient ;
  - la calibration porte sur la LOI des requetes (isotrope, echelle estimee sur
    les donnees), et non sur un point q0 particulier. C'est ce qui distingue EQS
    de l'expansion query-centered que COBS a testee et trouvee nuisible : on ne
    linearise pas autour d'un point, on ajuste une famille exacte globalement.
  - cout : r scalaires de plus par bloc (les poids remplacent les effectifs).

================================================================================
CONCEPT 2 -- Selection CERTIFIEE
================================================================================
La borne (*) de summaries.py est BILATERALE :
        | l_b(q) - lhat_b(q) |  <=  delta_b := s ||q|| rho_b
On dispose donc, GRATUITEMENT, d'un encadrement de la vraie log-masse de chaque
bloc :  L_b = lhat_b - delta_b  <=  l_b  <=  lhat_b + delta_b = U_b.

Cela permet une selection avec CERTIFICAT, ce qu'aucun selecteur actuel n'offre :
  1. tau = k-ieme plus grande borne INFERIEURE
  2. tout bloc dont la borne SUPERIEURE U_b < tau est PROUVE hors du top-k
  3. l'ensemble ambigu A = {b : U_b >= tau} contient donc forcement le vrai top-k
  4. si |A| = k, la selection est PROUVEE exacte, sans avoir lu une seule cle
  5. sinon, il suffit de raffiner les |A| blocs ambigus (lecture exacte)

Le cout devient DEPENDANT DES DONNEES au lieu d'etre fixe, et la qualite devient
GARANTIE au lieu d'etre esperee. C'est un changement de nature du probleme :
on passe d'une heuristique a un algorithme de separation-evaluation.

Raffinement anisotrope. La borne s||q||rho est le pire cas sur la direction. En
stockant en plus la direction principale u_c du cluster et son extension a_c, on
obtient une borne plus serree par Cauchy-Schwarz anisotrope :
        |q.delta| <= |q.u_c| a_c + ||q_perp|| rho_perp,c
"""
import numpy as np
import summaries as S


# ==============================================================================
# CONCEPT 1 : EQS
# ==============================================================================
def s_eqs(K, r, n_cal=256, sigma=None, how="kcenter", seed=0, ridge=1e-3):
    """
    Resume par quadrature exponentielle : r centroides + r poids AJUSTES.

    Les poids resolvent  min_w  sum_i ( (sum_c w_c e^{s q_i.z_c}) / y_i - 1 )^2
    ou y_i est la masse exacte du bloc pour la requete de calibration q_i.
    C'est un moindres carres lineaire en w (masse lineaire en w), regularise.
    """
    L, D = K.shape
    s = 1.0 / np.sqrt(D)
    lab = (S.kcenter if how == "kcenter" else S.kmeans)(K, r)
    Z, n = [], []
    for j in np.unique(lab):
        m = lab == j
        Z.append(K[m].mean(0)); n.append(m.sum())
    Z = np.stack(Z); n = np.array(n, float)

    if sigma is None:
        sigma = 1.0
    rng = np.random.default_rng(seed)
    Qc = rng.normal(scale=sigma, size=(n_cal, D)).astype(np.float32)

    # cibles et matrice de conception, en echelle relative
    ly = S.true_logmass(K, Qc, s)                       # log y_i
    A = np.exp(s * (Qc @ Z.T) - ly[:, None])            # (n_cal, r), deja divise par y_i
    b = np.ones(n_cal)
    # moindres carres regularise vers les effectifs (repli sur le coreset nu)
    w0 = n / n.sum() * 0 + n                            # depart : effectifs
    G = A.T @ A + ridge * np.eye(len(Z)) * np.trace(A.T @ A) / max(len(Z), 1)
    rhs = A.T @ b + ridge * np.trace(A.T @ A) / max(len(Z), 1) * w0
    try:
        w = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        w = w0
    w = np.maximum(w, 1e-6)
    return dict(kind="coreset", mu=Z, n=w, rho=S.max_radius(K, lab)), \
        len(Z) * (1.0 + 1.0 / D)


def calib_sigma(Q):
    """Echelle isotrope equivalente de la loi des requetes."""
    return float(np.sqrt((Q ** 2).sum(1).mean() / Q.shape[1]))


# ==============================================================================
# CONCEPT 2 : selection certifiee
# ==============================================================================
def certified_select(scores, rho, Qnorm, s, k):
    """
    scores : (nq, nb) log-masses estimees
    rho    : (nb,)    rayon max de la partition de chaque bloc
    Qnorm  : (nq,)    norme des requetes
    Retourne (ambig, tau) avec ambig (nq, nb) booleen : blocs non elimines.
    """
    delta = s * Qnorm[:, None] * rho[None, :]            # (nq, nb)
    Lo, Up = scores - delta, scores + delta
    # seuil : k-ieme plus grande borne inferieure
    tau = np.sort(Lo, 1)[:, -k][:, None]
    return Up >= tau, tau


def anisotropic_delta(K, lab, Q, s):
    """
    Borne anisotrope : au lieu de ||q|| rho, on decompose l'ecart intra-cluster
    sur sa direction principale. Renvoie delta (nq,) pour un bloc.
        |q.d| <= |q.u| a + ||q_perp|| rho_perp
    """
    best = None
    for j in np.unique(lab):
        m = lab == j
        Kc = K[m]
        mu = Kc.mean(0)
        Dl = Kc - mu
        if Dl.shape[0] < 2:
            d = np.zeros(Q.shape[0])
        else:
            U, sv, Vt = np.linalg.svd(Dl, full_matrices=False)
            u = Vt[0]
            a = float(np.abs(Dl @ u).max())
            Rperp = Dl - np.outer(Dl @ u, u)
            rp = float(np.linalg.norm(Rperp, axis=1).max())
            qu = np.abs(Q @ u)
            qperp = np.sqrt(np.maximum((Q ** 2).sum(1) - (Q @ u) ** 2, 0))
            d = qu * a + qperp * rp
        best = d if best is None else np.maximum(best, d)
    return s * best
