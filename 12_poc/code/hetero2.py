"""
FRONT 2, suite — test de permutation : structure en rang, ou forme de la distribution ?

Deux explications concurrentes des +5 a +8 points, qu'il faut separer proprement :

  (H1) STRUCTURE EN RANG. L'erreur depend systematiquement du rang vrai du bloc
       (heteroscedasticite, biais differentiel). Elle serait alors "bien placee".

  (H2) FORME MARGINALE. L'erreur n'est pas gaussienne mais a QUEUE LOURDE : la plupart
       des blocs ont une erreur faible, quelques-uns une erreur enorme. A ecart-type
       egal, une telle loi classe MIEUX qu'une gaussienne, car le classement ne depend
       que de la masse typique de l'erreur, pas de sa variance. C'est plausible pour
       COBS : son erreur est la part NON RETENUE de 1/2 q^T Sigma q, une somme de
       carres — donc de type chi-2, fortement asymetrique et leptokurtique.

TEST DÉCISIF — permutation intra-requete. On compare trois classements :
  (i)  erreur REELLE, telle quelle                      -> rappel observe
  (ii) erreur reelle PERMUTEE au hasard entre blocs, a l'interieur de chaque requete.
       Cela DETRUIT toute relation entre erreur et rang, mais CONSERVE exactement la
       distribution marginale (donc la forme, les queues, la variance).
  (iii) bruit gaussien iid de meme ecart-type par requete -> la loi actuelle

Lecture :
  (ii) ~ (iii) < (i)  ->  H1 : c'est la structure en rang qui explique l'ecart
  (ii) ~ (i)  > (iii) ->  H2 : c'est la forme de la loi (queues lourdes)
  intermediaire       ->  les deux contribuent, dans les proportions mesurees

On mesure en plus l'exces de kurtosis de l'erreur, qui doit predire l'ecart si H2.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"
KTOP = 8
SIGREF = np.array([0.0, .05, .1, .2, .35, .5, .75, 1., 1.5, 2., 3., 5.])
METHODS = {
    "mean":        lambda K: S.s_mean(K),
    "cobs r=2":    lambda K: S.s_cobs(K, 2),
    "cobs r=4":    lambda K: S.s_cobs(K, 4),
    "cobs r=8":    lambda K: S.s_cobs(K, 8),
    "coreset r=2": lambda K: S.s_coreset(K, 2),
    "coreset r=4": lambda K: S.s_coreset(K, 4),
    "coreset r=8": lambda K: S.s_coreset(K, 8),
    "quest":       lambda K: S.s_quest(K),
    "gmm 2x1":     lambda K: S.s_gmm(K, 2, 1),
    "gmm 4x1":     lambda K: S.s_gmm(K, 4, 1),
}


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")
    ref = {x: [] for x in SIGREF}
    acc = {m: dict(obs=[], perm=[], iid=[], sig=[], kurt=[], mad=[]) for m in METHODS}

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

            def rec(sc):
                om = np.argsort(-sc, 1)[:, :KTOP]
                return float(np.mean(np.take_along_axis(mass, om, 1).sum(1)
                                     / np.maximum(ora, 1e-30)))

            for x in SIGREF:
                ref[x].append(rec(Tm + (rng.normal(scale=x, size=Tm.shape) if x > 0 else 0)))

            for m in METHODS:
                Sc = np.stack([S.score(sums[m][i], Q, s) for i in range(n)], 1)
                e = Sc - Tm
                eps = e - e.mean(1, keepdims=True)
                sig = eps.std(1, keepdims=True)
                # (ii) permutation intra-requete : conserve la loi marginale exactement
                perm = np.stack([rng.permutation(eps[i]) for i in range(len(Q))])
                # (iii) gaussien de meme sigma par requete
                gau = rng.normal(size=eps.shape) * sig
                acc[m]["obs"].append(rec(Tm + eps))
                acc[m]["perm"].append(rec(Tm + perm))
                acc[m]["iid"].append(rec(Tm + gau))
                acc[m]["sig"].append(float(sig.mean()))
                z = eps / np.maximum(sig, 1e-9)
                acc[m]["kurt"].append(float((z ** 4).mean() - 3.0))
                # ecart absolu median rapporte a sigma : petit = queues lourdes
                acc[m]["mad"].append(float(np.median(np.abs(z))))
    return acc, {float(x): float(np.mean(ref[x])) for x in SIGREF}


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        acc, refc = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        xs = np.array(sorted(refc)); ys = np.array([refc[x] for x in xs])
        print("=" * 108)
        print(f"TEST DE PERMUTATION — structure en rang ou forme de la loi ?   ({lbl})")
        print("=" * 108)
        print(f"{'methode':13s} {'sigma':>7s} {'kurtose':>8s} {'MAD/sig':>8s} | "
              f"{'(i) reel':>9s} {'(ii) permute':>13s} {'(iii) gaussien':>15s} | "
              f"{'i-ii':>6s} {'ii-iii':>7s}")
        print("-" * 108)
        rows = []
        for m in METHODS:
            o, pm, ii = (np.mean(acc[m][k]) for k in ("obs", "perm", "iid"))
            sg, ku, md = (np.mean(acc[m][k]) for k in ("sig", "kurt", "mad"))
            print(f"{m:13s} {sg:7.3f} {ku:8.2f} {md:8.3f} | {100*o:8.2f}% "
                  f"{100*pm:12.2f}% {100*ii:14.2f}% | {100*(o-pm):+5.2f} {100*(pm-ii):+6.2f}")
            rows.append(dict(m=m, sigma=sg, kurt=ku, mad=md, obs=o, perm=pm, iid=ii))
        d_rank = np.mean([100 * (r["obs"] - r["perm"]) for r in rows])
        d_shape = np.mean([100 * (r["perm"] - r["iid"]) for r in rows])
        print("-" * 108)
        print(f"  effet moyen de la STRUCTURE EN RANG  (i)-(ii)  : {d_rank:+5.2f} points")
        print(f"  effet moyen de la FORME DE LA LOI    (ii)-(iii): {d_shape:+5.2f} points")
        tot = abs(d_rank) + abs(d_shape)
        if tot > 0:
            print(f"  -> repartition : {100*abs(d_rank)/tot:.0f}% structure, "
                  f"{100*abs(d_shape)/tot:.0f}% forme")
        # correlation kurtose <-> ecart a la loi iid
        k = np.array([r["kurt"] for r in rows])
        gap = np.array([100 * (r["obs"] - float(np.interp(r["sigma"], xs, ys)))
                        for r in rows])
        cc = float(np.corrcoef(k, gap)[0, 1])
        print(f"\n  correlation (exces de kurtose, ecart a la loi iid) = {cc:+.3f}")
        print(f"  correlation (MAD/sigma,          ecart a la loi iid) = "
              f"{float(np.corrcoef([r['mad'] for r in rows], gap)[0,1]):+.3f}")
        out[lbl] = dict(rows=rows, d_rank=d_rank, d_shape=d_shape, corr_kurt=cc)
        print()
    (RES / "hetero2.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'hetero2.json'}")
