"""
LOI UNIVERSELLE DE SELECTION ?

Hypothese. Le rappel de masse d'un selecteur de blocs ne depend de lui que par une
seule grandeur : l'ecart-type, ENTRE BLOCS et A REQUETE FIXEE, de son erreur
d'estimation demoyennee

    eps_b(q) = [lhat_b(q) - l_b(q)] - moyenne_b[lhat_b(q) - l_b(q)]
    sigma_disc = E_q[ ecart-type_b( eps_b(q) ) ]

Justification. Ajouter une constante m(q) a tous les scores d'une requete ne change
aucun classement. Seule la composante VARIABLE ENTRE BLOCS discrimine. C'est
pourquoi EQS, qui corrige le mode commun, divise le RMSE par deux sans rien gagner
en rappel.

Protocole de test. On construit la courbe de reference par INJECTION DE BRUIT : on
part des log-masses EXACTES et on y ajoute un bruit gaussien d'ecart-type sigma
controle, puis on mesure le rappel. Cela donne rappel(sigma) independamment de
toute methode. On place ensuite chaque methode reelle a son sigma_disc mesure. Si
les points tombent sur la courbe, la loi tient.

Consequence si elle tient : le probleme de la conception d'un resume de bloc se
reduit a MINIMISER sigma_disc SOUS CONTRAINTE D'OCTETS, et la courbe donne
directement le plafond de rappel atteignable.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"
KS = (4, 8, 16)
SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

METHODS = {
    "mean":        lambda K: S.s_mean(K),
    "maxpool":     lambda K: S.s_maxpool(K),
    "quest":       lambda K: S.s_quest(K),
    "cobs r=1":    lambda K: S.s_cobs(K, 1),
    "cobs r=2":    lambda K: S.s_cobs(K, 2),
    "cobs r=4":    lambda K: S.s_cobs(K, 4),
    "cobs r=8":    lambda K: S.s_cobs(K, 8),
    "coreset r=2": lambda K: S.s_coreset(K, 2),
    "coreset r=4": lambda K: S.s_coreset(K, 4),
    "coreset r=8": lambda K: S.s_coreset(K, 8),
    "gmm 2x1":     lambda K: S.s_gmm(K, 2, 1),
    "gmm 4x1":     lambda K: S.s_gmm(K, 4, 1),
}


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    meth = {m: {"sig": [], "rmse": [], "sperp": [], "slope": [], **{k: [] for k in KS}} for m in METHODS}
    ref = {sg: {k: [] for k in KS} for sg in SIGMAS}
    cost = {}

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 20:
                continue
            cand = np.arange(1, hi)
            sums = {m: [] for m in METHODS}
            for b in cand:
                Kb = K[b * Lb:(b + 1) * Lb]
                for m, fn in METHODS.items():
                    su, c = fn(Kb)
                    sums[m].append(su); cost[m] = c
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                           for b in cand], 1)
            mass = np.exp(Tm - Tm.max(1, keepdims=True))
            order = np.argsort(-Tm, 1)

            def recall(Sc):
                om = np.argsort(-Sc, 1)
                out = {}
                for k in KS:
                    got = np.take_along_axis(mass, om[:, :k], 1).sum(1)
                    ora = np.take_along_axis(mass, order[:, :k], 1).sum(1)
                    out[k] = float(np.mean(got / np.maximum(ora, 1e-30)))
                return out

            # --- courbe de reference : bruit gaussien controle sur les vraies masses
            for sg in SIGMAS:
                noise = rng.normal(scale=sg, size=Tm.shape) if sg > 0 else 0.0
                r = recall(Tm + noise)
                for k in KS:
                    ref[sg][k].append(r[k])

            # --- methodes reelles
            for m in METHODS:
                Sc = np.stack([S.score(sums[m][i], Q, s) for i in range(len(cand))], 1)
                e = Sc - Tm
                eps = e - e.mean(1, keepdims=True)        # demoyennage PAR REQUETE
                meth[m]["sig"].append(float(eps.std(1).mean()))
                meth[m]["rmse"].append(float(np.sqrt((e ** 2).mean())))
                # --- decomposition : part de l'erreur COLINEAIRE au signal vrai
                # eps_b = a * (l_b - lbar) + eps_perp_b.  Seule eps_perp nuit au
                # classement : une erreur proportionnelle au signal ne fait que le
                # redimensionner, sans changer aucun ordre (si a > -1).
                sig_c = Tm - Tm.mean(1, keepdims=True)
                den = (sig_c ** 2).sum(1, keepdims=True)
                a = (eps * sig_c).sum(1, keepdims=True) / np.maximum(den, 1e-30)
                perp = eps - a * sig_c
                meth[m]["sperp"].append(float(perp.std(1).mean()))
                meth[m]["slope"].append(float(a.mean()))
                r = recall(Sc)
                for k in KS:
                    meth[m][k].append(r[k])
    return meth, ref, cost


def interp(ref_curve, sig):
    xs = np.array(sorted(ref_curve))
    ys = np.array([ref_curve[x] for x in xs])
    return float(np.interp(sig, xs, ys))


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        meth, ref, cost = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        refc = {k: {sg: float(np.mean(ref[sg][k])) for sg in SIGMAS} for k in KS}

        print("=" * 100)
        print(f"COURBE DE REFERENCE : rappel de masse en fonction du bruit injecte  ({lbl})")
        print("=" * 100)
        print(f"{'sigma (nats)':>13s} " + " ".join(f"{'@'+str(k):>9s}" for k in KS))
        print("-" * 100)
        for sg in SIGMAS:
            print(f"{sg:13.2f} " + " ".join(f"{100*refc[k][sg]:8.2f}%" for k in KS))

        print()
        print("=" * 100)
        print(f"LES METHODES REELLES TOMBENT-ELLES SUR LA COURBE ?  ({lbl})")
        print("=" * 100)
        print(f"{'methode':14s} {'cout':>5s} {'RMSE':>6s} {'sigma_disc':>11s} | "
              f"{'@8 mesure':>10s} {'@8 predit':>10s} {'ecart':>7s}")
        print("-" * 100)
        errs, errs2 = [], []
        rows = sorted(meth.items(), key=lambda kv: np.mean(kv[1]["sperp"]))
        print(f"{'':14s} {'':5s} {'':6s} {'':11s} | {'':10s} {'sigma_disc':>10s} "
              f"{'sigma_perp':>10s}")
        for m, d in rows:
            sg = float(np.mean(d["sig"])); sp = float(np.mean(d["sperp"]))
            obs = float(np.mean(d[8]))
            p1, p2 = interp(refc[8], sg), interp(refc[8], sp)
            errs.append(abs(obs - p1)); errs2.append(abs(obs - p2))
            print(f"{m:14s} {cost[m]:5.2f} {np.mean(d['rmse']):6.2f} "
                  f"a={np.mean(d['slope']):+7.3f} | {100*obs:9.2f}% "
                  f"{100*(obs-p1):+9.2f}% {100*(obs-p2):+9.2f}%")
        mae = float(np.mean(errs)); mae2 = float(np.mean(errs2))
        print("")
        print(f"  ecart de prediction avec sigma_disc : {100*mae:.2f} points")
        print(f"  ecart de prediction avec sigma_perp : {100*mae2:.2f} points"
              f"   ({'MEILLEUR' if mae2 < mae else 'pas mieux'})")
        if mae2 < 0.03:
            print("  -> LA LOI TIENT : sigma_perp suffit a predire le rappel a ~"
                  f"{100*mae2:.1f} points pres, quelle que soit la construction.")
        else:
            print("  -> la loi ne tient pas : d'autres proprietes du bruit comptent.")
        out[lbl] = dict(ref={str(k): refc[k] for k in KS},
                        methods={m: dict(cost=cost[m], sigma=float(np.mean(d["sig"])),
                                         rmse=float(np.mean(d["rmse"])),
                                         sperp=float(np.mean(d["sperp"])),
                                         slope=float(np.mean(d["slope"])),
                                         **{f"r@{k}": float(np.mean(d[k])) for k in KS})
                                 for m, d in meth.items()}, mae=mae, mae_perp=mae2)
        print()
    (RES / "law_selection.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'law_selection.json'}")
