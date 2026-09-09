"""
ASP — formalisation et vérification prédictive.

PROBLÈME D'OPTIMISATION
-----------------------
[DÉRIVATION] Notations. n blocs candidats ; on veut les k de plus grande log-masse
l_b. Un résumé de b octets donne une estimation de bruit sigma(b) (courbe
débit-distorsion, mesurée). Schéma à deux passes : b1 octets pour les n blocs,
b2 octets pour les m survivants. Coût total

    B(b1, b2, m) = n * b1 + m * b2

Perte de rappel. Elle a deux sources indépendantes.

(1) Élimination à tort en passe 1. Un bloc de rang vrai j <= k sort du top-m si son
    score bruité passe sous celui du bloc de rang m. Les deux scores étant bruités
    indépendamment, la différence a pour écart-type sigma1*sqrt(2), d'où

        P(perdre le rang j) = Phi( -Delta_{j,m} / (sigma1 * sqrt(2)) ),
        Delta_{j,m} = l_(j) - l_(m)

    La masse perdue est ponderée par la part de masse w_j du rang j :

        L1(b1, m) = somme_{j<=k} w_j * Phi( -Delta_{j,m} / (sigma(b1) sqrt(2)) )

(2) Erreur de classement en passe 2, donnée directement par la courbe de référence
    mesurée dans law_selection.py :

        L2(b2) = 1 - R(sigma(b2))

Le rappel attendu vaut donc, au premier ordre (les deux pertes étant sur des
populations de blocs disjointes) :

        Rappel(b1, b2, m)  ~  R(sigma(b2)) - L1(b1, m)

et le problème est

        min  n*b1 + m*b2      sous contrainte    Rappel(b1,b2,m) >= cible

Toutes les grandeurs (sigma(b), Delta_{j,m}, w_j, R) sont MESURABLES hors ligne sur
un échantillon de calibration. On peut donc résoudre le problème sans balayage
empirique. C'est ce qu'on vérifie ici : la configuration prédite coïncide-t-elle
avec l'optimum observé ?
"""
import numpy as np, json, pathlib
from scipy.stats import norm
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def measure(tag, use_rope, Lb=64, nq=48, local=8, topk=8, seed=0,
            rs=(1, 2, 4, 8, 16, 32)):
    """Mesure sigma(r), Delta_{j,m}, w_j et la courbe de reference R(sigma)."""
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    sig = {r: [] for r in rs}
    cost = {}
    W = []                       # parts de masse des rangs 1..k
    GAP = []                     # l_(j) - l_(m) pour j<=k, m dans MS
    MS = np.array([10, 12, 16, 20, 24, 32, 40, 48])
    SIGREF = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    REF = {sg: [] for sg in SIGREF}
    n_obs = []

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 56:
                continue
            cand = np.arange(1, hi); n = len(cand); n_obs.append(n)
            sums = {}
            for r in rs:
                sums[r] = []
                for b in cand:
                    su, c = S.s_coreset(K[b * Lb:(b + 1) * Lb], r)
                    sums[r].append(su); cost[r] = c
            Q = Q_all[qpos, kv * g, :].astype(np.float32)
            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                           for b in cand], 1)
            mass = np.exp(Tm - Tm.max(1, keepdims=True))
            order = np.argsort(-Tm, 1)
            srt = np.sort(Tm, 1)[:, ::-1]
            msrt = np.take_along_axis(mass, order, 1)
            W.append((msrt[:, :topk] / msrt[:, :topk].sum(1, keepdims=True)).mean(0))
            GAP.append(np.stack([[np.mean(srt[:, j] - srt[:, m - 1]) for m in MS]
                                 for j in range(topk)]))
            for r in rs:
                Sc = np.stack([S.score(sums[r][b], Q, s) for b in range(n)], 1)
                e = Sc - Tm
                eps = e - e.mean(1, keepdims=True)
                sig[r].append(float(eps.std(1).mean()))
            ora = np.take_along_axis(mass, order[:, :topk], 1).sum(1)
            for sg in SIGREF:
                sc = Tm + (rng.normal(scale=sg, size=Tm.shape) if sg > 0 else 0)
                om = np.argsort(-sc, 1)[:, :topk]
                got = np.take_along_axis(mass, om, 1).sum(1)
                REF[sg].append(float(np.mean(got / np.maximum(ora, 1e-30))))
    return (dict(sigma={r: float(np.mean(v)) for r, v in sig.items()},
                 cost={r: cost[r] for r in rs},
                 w=np.mean(W, 0), gap=np.mean(GAP, 0), MS=MS,
                 sigref=SIGREF, ref=np.array([np.mean(REF[s_]) for s_ in SIGREF]),
                 n=float(np.mean(n_obs)), topk=topk))


def predict(M, r1, r2, m):
    """Rappel prédit et coût prédit d'une configuration ASP."""
    n, k = M["n"], M["topk"]
    s1 = M["sigma"][r1]
    R2 = float(np.interp(M["sigma"][r2], M["sigref"], M["ref"]))
    im = int(np.argmin(np.abs(M["MS"] - m)))
    L1 = float(np.sum(M["w"] * norm.cdf(-M["gap"][:, im] / (s1 * np.sqrt(2)))))
    return R2 - L1, M["cost"][r1] * n + M["cost"][r2] * m


if __name__ == "__main__":
    out = {}
    for tag in ("qwen8k", "smol8k"):
        for rope in (True,):
            M = measure(tag, rope)
            lbl = f"{tag} {'avec' if rope else 'sans'} RoPE"
            print("=" * 100)
            print(f"ASP — le modèle prédit-il les bonnes configurations ?   ({lbl})")
            print("=" * 100)
            print(f"  n = {M['n']:.0f} blocs, k = {M['topk']}")
            print(f"  sigma(r) mesuré : " +
                  "  ".join(f"r={r}:{v:.3f}" for r, v in sorted(M["sigma"].items())))
            print(f"  parts de masse des rangs 1..{M['topk']} : "
                  + " ".join(f"{x:.3f}" for x in M["w"]))
            print()
            print(f"  {'config':16s} {'coût':>9s} {'rappel prédit':>14s}")
            print("  " + "-" * 60)
            cands = []
            for r1 in (1, 2, 4):
                for r2 in (4, 8, 16, 32):
                    if r2 <= r1:
                        continue
                    for m in M["MS"]:
                        rec, b = predict(M, r1, r2, int(m))
                        cands.append((b, rec, r1, r2, int(m)))
            # front de Pareto prédit
            cands.sort(key=lambda z: z[0])
            best = -1
            front = []
            for b, rec, r1, r2, m in cands:
                if rec > best:
                    best = rec; front.append((b, rec, r1, r2, m))
            for b, rec, r1, r2, m in front[:12]:
                print(f"  r1={r1},r2={r2:2d},m={m:2d}   {b:9.1f} {100*rec:13.2f}%")
            # comparaison au plat
            print(f"\n  {'plat':16s} {'coût':>9s} {'rappel prédit':>14s}")
            print("  " + "-" * 60)
            for r in sorted(M["sigma"]):
                R = float(np.interp(M["sigma"][r], M["sigref"], M["ref"]))
                print(f"  coreset r={r:<2d}       {M['cost'][r]*M['n']:9.1f} "
                      f"{100*R:13.2f}%")
            # gain prédit
            fb = np.array([M["cost"][r] * M["n"] for r in sorted(M["sigma"])])
            fr = np.array([float(np.interp(M["sigma"][r], M["sigref"], M["ref"]))
                           for r in sorted(M["sigma"])])
            gains = []
            for b, rec, r1, r2, m in front:
                if fr.min() < rec < fr.max():
                    gains.append(float(np.interp(rec, fr, fb)) / b)
            if gains:
                print(f"\n  gain PRÉDIT par le modèle : médian {np.median(gains):.2f}x, "
                      f"max {max(gains):.2f}x")
                print(f"  (gain MESURÉ en §asp_pareto : médian 1,24-1,64x, max 1,7-2,4x)")
            out[lbl] = dict(front=[[float(b), float(r), r1, r2, m]
                                   for b, r, r1, r2, m in front],
                            gains_pred=gains)
            print()
    (RES / "asp_optimizer.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"-> {RES/'asp_optimizer.json'}")
