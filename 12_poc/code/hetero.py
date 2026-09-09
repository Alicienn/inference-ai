"""
FRONT 2 — élucider les +5 à +8 points d'écart de la loi rappel(sigma).

RAPPEL DU PROBLÈME. La courbe de référence rappel(sigma), construite en injectant un
bruit gaussien iid sur les log-masses exactes, prédit les coresets à ±0,4 point mais
sous-estime COBS et GMM de +5 à +8 points : ils font BEAUCOUP mieux que ce que leur
sigma laisse attendre. L'hypothèse « erreur colinéaire au signal » a été REJETÉE
(pentes mesurées quasi nulles, -0,27 à +0,15).

HYPOTHÈSE À TESTER MAINTENANT, formulée pour être quantitative.
Le bruit de référence est iid : même loi pour tous les blocs, indépendante du rang.
Le bruit de COBS ne l'est pas. Son terme quadratique 1/2 q^T Sigma_b q est
  (a) TOUJOURS POSITIF, et
  (b) d'amplitude croissante avec la dispersion du bloc.
Si les blocs de fort rang (forte masse) sont aussi les plus dispersés, alors COBS les
sur-score SYSTÉMATIQUEMENT par rapport aux autres. Cela revient à AGRANDIR l'écart
entre le top-k et le reste — donc à améliorer le classement, sans que cela apparaisse
dans un sigma global ni dans une régression linéaire (la relation n'est pas linéaire
en l_b : elle est concentrée sur les premiers rangs).

FORMULATION TESTABLE. On caractérise l'erreur non par un scalaire mais par son PROFIL
EN RANG : pour chaque tranche de rang vrai, la moyenne mu_j et l'écart-type s_j de
l'erreur. Puis on construit un modèle génératif structuré

    eps_b = mu_{j(b)} + N(0, s_{j(b)}^2)        j(b) = tranche de rang de b

et on prédit le rappel par simulation avec CE bruit-là. Trois prédictions comparées :
  - iid global  : eps ~ N(0, sigma_all^2)          <- la loi actuelle, qui échoue
  - profil complet : le modèle ci-dessus
  - profil sans biais : mu_j forcés à 0, seuls les s_j conservés
Le troisième isole la part due à l'HÉTÉROSCÉDASTICITÉ SEULE ; l'écart entre le second
et le troisième isole la part due au BIAIS DIFFÉRENTIEL PAR RANG.

C'est un test décisif : si « profil complet » reproduit le rappel observé mais pas
« profil sans biais », l'explication n'est pas l'hétéroscédasticité mais le biais.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"
KTOP = 8
# tranches de rang vrai (bornes supérieures)
BINS = [1, 2, 4, 8, 16, 32, 64, 10**9]

METHODS = {
    "mean":        lambda K: S.s_mean(K),
    "cobs r=2":    lambda K: S.s_cobs(K, 2),
    "cobs r=4":    lambda K: S.s_cobs(K, 4),
    "cobs r=8":    lambda K: S.s_cobs(K, 8),
    "coreset r=2": lambda K: S.s_coreset(K, 2),
    "coreset r=4": lambda K: S.s_coreset(K, 4),
    "coreset r=8": lambda K: S.s_coreset(K, 8),
    "gmm 2x1":     lambda K: S.s_gmm(K, 2, 1),
    "gmm 4x1":     lambda K: S.s_gmm(K, 4, 1),
}
SIGREF = np.array([0.0, .05, .1, .2, .35, .5, .75, 1., 1.5, 2., 3., 5.])


def binid(ranks):
    out = np.zeros_like(ranks)
    for i, b in enumerate(BINS):
        out[(ranks < b) & (out == 0) & (ranks >= (BINS[i - 1] if i else 0))] = i
    return np.clip(out, 0, len(BINS) - 1)


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    prof = {m: {j: [] for j in range(len(BINS))} for m in METHODS}
    obs = {m: [] for m in METHODS}
    sig_all = {m: [] for m in METHODS}
    ref = {sg: [] for sg in SIGREF}
    sim = {m: dict(full=[], nobias=[], iid=[]) for m in METHODS}

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 40:
                continue
            cand = np.arange(1, hi); n = len(cand)
            sums = {m: [fn(K[b * Lb:(b + 1) * Lb])[0] for b in cand]
                    for m, fn in METHODS.items()}
            Q = Q_all[qpos, kv * g, :].astype(np.float32)
            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                           for b in cand], 1)
            mass = np.exp(Tm - Tm.max(1, keepdims=True))
            order = np.argsort(-Tm, 1)
            ora = np.take_along_axis(mass, order[:, :KTOP], 1).sum(1)
            # rang vrai de chaque bloc
            rank = np.empty_like(order)
            np.put_along_axis(rank, order, np.arange(n)[None, :].repeat(len(Q), 0), 1)
            bid = binid(rank)

            def recall(sc):
                om = np.argsort(-sc, 1)[:, :KTOP]
                return float(np.mean(np.take_along_axis(mass, om, 1).sum(1)
                                     / np.maximum(ora, 1e-30)))

            for sg in SIGREF:
                ref[sg].append(recall(Tm + (rng.normal(scale=sg, size=Tm.shape)
                                            if sg > 0 else 0)))

            for m in METHODS:
                Sc = np.stack([S.score(sums[m][i], Q, s) for i in range(n)], 1)
                e = Sc - Tm
                eps = e - e.mean(1, keepdims=True)     # demoyennage par requete
                obs[m].append(recall(Sc))
                sig_all[m].append(float(eps.std(1).mean()))
                for j in range(len(BINS)):
                    sel = bid == j
                    if sel.any():
                        prof[m][j].append((float(eps[sel].mean()),
                                           float(eps[sel].std()), int(sel.sum())))
                # --- simulations avec bruit structure, reconstruit sur place
                mu = np.zeros(len(BINS)); sd = np.zeros(len(BINS))
                for j in range(len(BINS)):
                    sel = bid == j
                    if sel.any():
                        mu[j] = eps[sel].mean(); sd[j] = eps[sel].std()
                Efull = mu[bid] + rng.normal(size=bid.shape) * sd[bid]
                Enob = rng.normal(size=bid.shape) * sd[bid]
                Eiid = rng.normal(size=bid.shape) * float(eps.std(1).mean())
                sim[m]["full"].append(recall(Tm + Efull))
                sim[m]["nobias"].append(recall(Tm + Enob))
                sim[m]["iid"].append(recall(Tm + Eiid))
    refc = {float(sg): float(np.mean(ref[sg])) for sg in SIGREF}
    return prof, obs, sig_all, refc, sim


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        prof, obs, sig_all, refc, sim = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        xs = np.array(sorted(refc)); ys = np.array([refc[x] for x in xs])

        print("=" * 104)
        print(f"PROFIL EN RANG DE L'ERREUR  ({lbl}, top-8)  — moyenne / ecart-type par tranche")
        print("=" * 104)
        hdr = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]
        print(f"{'methode':13s} " + " ".join(f"{h:>11s}" for h in hdr))
        print("-" * 104)
        for m in METHODS:
            cells = []
            for j in range(len(BINS)):
                v = prof[m][j]
                cells.append(f"{np.mean([x[0] for x in v]):+5.2f}/{np.mean([x[1] for x in v]):4.2f}"
                             if v else "     -     ")
            print(f"{m:13s} " + " ".join(f"{c:>11s}" for c in cells))
        print("\n  lecture : 'moyenne/ecart-type' de l'erreur demoyennee, par tranche de rang vrai.")
        print("  Un biais POSITIF sur les premiers rangs = la methode sur-score les blocs")
        print("  qui comptent, ce qui AGRANDIT l'ecart et ameliore le classement.")

        print()
        print("=" * 104)
        print(f"TEST DÉCISIF : quel modèle de bruit reproduit le rappel observé ?  ({lbl})")
        print("=" * 104)
        print(f"{'methode':13s} {'sigma_all':>9s} {'observe':>9s} | {'loi iid':>9s} "
              f"{'ecart':>7s} | {'profil complet':>15s} {'ecart':>7s} | "
              f"{'profil sans biais':>18s} {'ecart':>7s}")
        print("-" * 104)
        e_iid, e_full, e_nob = [], [], []
        for m in METHODS:
            o = np.mean(obs[m]); sa = np.mean(sig_all[m])
            p_iid = float(np.interp(sa, xs, ys))
            p_full, p_nob = np.mean(sim[m]["full"]), np.mean(sim[m]["nobias"])
            e_iid.append(o - p_iid); e_full.append(o - p_full); e_nob.append(o - p_nob)
            print(f"{m:13s} {sa:9.3f} {100*o:8.2f}% | {100*p_iid:8.2f}% "
                  f"{100*(o-p_iid):+6.2f} | {100*p_full:14.2f}% {100*(o-p_full):+6.2f} | "
                  f"{100*p_nob:17.2f}% {100*(o-p_nob):+6.2f}")
        for nm, e in (("loi iid", e_iid), ("profil complet", e_full),
                      ("profil sans biais", e_nob)):
            print(f"  ecart absolu moyen, {nm:18s} : {100*np.mean(np.abs(e)):5.2f} points")
        out[lbl] = dict(
            profile={m: {str(j): [float(np.mean([x[0] for x in prof[m][j]])),
                                  float(np.mean([x[1] for x in prof[m][j]]))]
                         for j in range(len(BINS)) if prof[m][j]} for m in METHODS},
            table={m: dict(sigma=float(np.mean(sig_all[m])), obs=float(np.mean(obs[m])),
                           iid=float(np.interp(np.mean(sig_all[m]), xs, ys)),
                           full=float(np.mean(sim[m]["full"])),
                           nobias=float(np.mean(sim[m]["nobias"]))) for m in METHODS},
            mae=dict(iid=float(np.mean(np.abs(e_iid))), full=float(np.mean(np.abs(e_full))),
                     nobias=float(np.mean(np.abs(e_nob)))))
        print()
    (RES / "hetero.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'hetero.json'}")
