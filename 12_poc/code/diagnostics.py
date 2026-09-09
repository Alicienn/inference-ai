"""
Diagnostics : POURQUOI le terme de covariance echoue-t-il ?

Hypothese a tester. Le developpement en cumulants tronque a l'ordre 2 n'est
valide que si le PARAMETRE DE DEVELOPPEMENT est petit :

    eps_2 = (s^2/2) q^T Sigma_b q        (terme d'ordre 2, en nats)

Si eps_2 est d'ordre 1 ou plus, la troncature n'est pas une petite correction :
la serie est asymptotique et la tronquer a un ordre fini n'offre aucune garantie.
On mesure ici eps_2 sur donnees reelles, ainsi que le RESTE reel

    R = l_b(q) - [log L + s q.mu + (s^2/2) q^T Sigma q]

Si |R| >> |eps_2|, le second ordre ne capture qu'une fraction negligeable de
l'ecart et l'on est bien en regime de GRANDES DEVIATIONS.

On mesure aussi l'ecart de la borne coreset (*) pour verifier qu'elle tient et
qu'elle est informative.
"""
import numpy as np, json, pathlib, collections
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def run(tag="qwen8k", Lb=64, nq=32, local=8, use_rope=True, seed=0):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)

    eps1, eps2, rest, ratio = [], [], [], []
    bound_ok, bound_tight = [], []
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            nb = K.shape[0] // Lb
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 8:
                continue
            for h in range(kv * g, (kv + 1) * g, max(1, g // 2)):
                Q = Q_all[qpos, h, :].astype(np.float32)
                for b in range(1, hi):
                    Kb = K[b * Lb:(b + 1) * Lb]
                    mu = Kb.mean(0)
                    X = Kb - mu
                    Sig = X.T @ X / Lb
                    t = S.true_logmass(Kb, Q, s) - np.log(Lb)
                    e1 = s * (Q @ mu)
                    e2 = 0.5 * s * s * np.einsum("qd,de,qe->q", Q, Sig, Q)
                    R = t - e1 - e2
                    eps1.append(np.abs(e1).mean()); eps2.append(e2.mean())
                    rest.append(np.abs(R).mean())
                    ratio.append((np.abs(R) / np.maximum(e2, 1e-9)).mean())
                    # borne coreset
                    su, _ = S.s_coreset(Kb, 4)
                    err = np.abs(S.score(su, Q, s) - S.true_logmass(Kb, Q, s))
                    bd = s * np.linalg.norm(Q, axis=1) * su["rho"]
                    bound_ok.append(float((err <= bd + 1e-4).mean()))
                    bound_tight.append(float((err / np.maximum(bd, 1e-9)).mean()))
    return (np.array(eps1), np.array(eps2), np.array(rest), np.array(ratio),
            np.array(bound_ok), np.array(bound_tight))


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        e1, e2, R, rat, bok, btight = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        print("=" * 96)
        print(f"REGIME DE DEVELOPPEMENT  --  {lbl}  (bloc L=64, Qwen2.5-0.5B)")
        print("=" * 96)
        print(f"  |terme d'ordre 1|  moyen : {e1.mean():8.3f} nats")
        print(f"   terme d'ordre 2   moyen : {e2.mean():8.3f} nats   "
              f"(median {np.median(e2):.3f}, p90 {np.percentile(e2,90):.3f})")
        print(f"  |reste apres ordre 2|    : {R.mean():8.3f} nats   "
              f"(median {np.median(R):.3f}, p90 {np.percentile(R,90):.3f})")
        print(f"\n  ratio |reste| / |ordre 2| : moyen {rat.mean():6.2f}   "
              f"median {np.median(rat):6.2f}")
        if np.median(rat) > 1:
            print("  -> le RESTE domine la correction d'ordre 2 : la troncature ne")
            print("     capture qu'une fraction de l'ecart. Regime de GRANDES DEVIATIONS,")
            print("     hors du domaine ou le developpement en cumulants est fiable.")
        else:
            print("  -> l'ordre 2 capture l'essentiel : regime de petites deviations.")
        print(f"\n  borne coreset (*) : respectee dans {100*bok.mean():.2f}% des cas, "
              f"serrement moyen {btight.mean():.3f}")
        out[lbl] = dict(ordre1=float(e1.mean()), ordre2=float(e2.mean()),
                        reste=float(R.mean()), ratio_median=float(np.median(rat)),
                        borne_respectee=float(bok.mean()),
                        borne_serrement=float(btight.mean()))
        print()
    (RES / "diagnostics_regime.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'diagnostics_regime.json'}")
