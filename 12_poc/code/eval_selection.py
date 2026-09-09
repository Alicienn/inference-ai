"""
Banc d'essai : qualite de la selection de blocs sur Q/K REELS (Qwen2.5-0.5B, 8k).

PROTOCOLE. Dans un systeme bloc-creux reel (NSA, CSA, COBS), la fenetre locale et
le puits d'attention (bloc 0) sont conserves par des branches SEPAREES. La branche
de selection n'a donc a retrouver que les blocs DISTANTS. On exclut donc des
candidats le bloc 0 et les `local` blocs precedant la requete : sans cela, la
metrique est dominee par des blocs triviaux et ne mesure plus la recuperation.

On compare, A BUDGET D'OCTETS EGAL, les estimateurs du log-masse d'un bloc :
  - mean-pool     premier ordre (base NSA)
  - Quest         boite englobante min/max : approximation grossiere du SUPPORT
  - COBS r        moyenne + covariance rang r : approximation par MOMENTS
  - coreset r     r centroides + effectifs : approximation du SUPPORT      [nous]
  - coreset+var   idem + variance isotrope par cluster (1 scalaire)        [nous]
  - coreset+rad   idem + borne de rayon (optimisme principiel)             [nous]
  - gmm c x rk    melange gaussien : famille unifiee                       [nous]

Metrique principale : RAPPEL DE MASSE @k = masse d'attention vraie captee par les
k blocs selectionnes / masse des k meilleurs blocs. C'est ce qui determine la
qualite finale de l'attention creuse.
"""
import numpy as np, json, pathlib, argparse, collections
import summaries as S

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"

METHODS = {
    "mean":            lambda K: S.s_mean(K),
    "maxpool":         lambda K: S.s_maxpool(K),
    "quest":           lambda K: S.s_quest(K),
    "cobs r=1":        lambda K: S.s_cobs(K, 1),
    "cobs r=2":        lambda K: S.s_cobs(K, 2),
    "cobs r=4":        lambda K: S.s_cobs(K, 4),
    "cobs r=8":        lambda K: S.s_cobs(K, 8),
    "coreset r=2":     lambda K: S.s_coreset(K, 2),
    "coreset r=4":     lambda K: S.s_coreset(K, 4),
    "coreset r=8":     lambda K: S.s_coreset(K, 8),
    "coreset-km r=4":  lambda K: S.s_coreset(K, 4, "kmeans"),
    "cs+var r=2":      lambda K: S.s_coreset_var(K, 2),
    "cs+var r=4":      lambda K: S.s_coreset_var(K, 4),
    "cs+var r=8":      lambda K: S.s_coreset_var(K, 8),
    "cs+rad r=4":      lambda K: S.s_coreset_rad(K, 4, 1.0),
    "cs+rad.5 r=4":    lambda K: S.s_coreset_rad(K, 4, 0.5),
    "cs+rad.5 r=8":    lambda K: S.s_coreset_rad(K, 8, 0.5),
    "gmm 2x1":         lambda K: S.s_gmm(K, 2, 1),
    "gmm 4x1":         lambda K: S.s_gmm(K, 4, 1),
}
KS = (1, 2, 4, 8, 16)


def load(tag):
    z = np.load(RES / f"qk_{tag}.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    packs = collections.defaultdict(dict)
    for k in z.files:
        if k == "meta":
            continue
        doc, layer, what = k.split("_", 2)
        packs[(int(doc), int(layer))][what] = z[k]
    return meta, packs


def evaluate(tag, Lb=64, nq=48, local=8, use_rope=True, seed=0, methods=None):
    methods = methods or METHODS
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)

    acc = {m: {k: [] for k in KS} for m in methods}
    for m in methods:
        acc[m]["rmse"] = []; acc[m]["bias"] = []
    costs, conc, needle_frac = {}, [], []

    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            nb_tot = K.shape[0] // Lb
            sums = {m: [] for m in methods}
            for b in range(nb_tot):
                Kb = K[b * Lb:(b + 1) * Lb]
                for m, fn in methods.items():
                    su, c = fn(Kb)
                    sums[m].append(su); costs[m] = c

            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            # candidats : on exclut le puits (bloc 0) et les `local` blocs recents
            hi = int(qpos.min()) // Lb - local
            if hi < max(KS) + 4:
                continue
            cand = np.arange(1, hi)

            for h in range(kv * g, (kv + 1) * g, max(1, g // 2)):
                Q = Q_all[qpos, h, :].astype(np.float32)
                Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                               for b in cand], 1)                    # (nq, nc)
                mass = np.exp(Tm - Tm.max(1, keepdims=True))
                order = np.argsort(-Tm, 1)
                share = np.take_along_axis(mass, order[:, :1], 1)[:, 0] / mass.sum(1)
                conc.append(float(share.mean()))
                needle_frac.append(float((share > 0.5).mean()))

                for m in methods:
                    Sc = np.stack([S.score(sums[m][b], Q, s) for b in cand], 1)
                    e = Sc - Tm
                    acc[m]["bias"].append(float(e.mean()))
                    acc[m]["rmse"].append(float(np.sqrt((e ** 2).mean())))
                    om = np.argsort(-Sc, 1)
                    for k in KS:
                        got = np.take_along_axis(mass, om[:, :k], 1).sum(1)
                        ora = np.take_along_axis(mass, order[:, :k], 1).sum(1)
                        acc[m][k].append(float(np.mean(got / np.maximum(ora, 1e-30))))
    return acc, costs, float(np.mean(conc)), float(np.mean(needle_frac))


def report(acc, costs, conc, nf, title):
    print("=" * 108)
    print(title)
    print("=" * 108)
    print(f"  part de masse du meilleur bloc : {conc:.3f}   "
          f"requetes 'aiguille' (>50% sur 1 bloc) : {100*nf:.1f}%")
    rows = sorted(acc.items(), key=lambda kv: (costs[kv[0]], kv[0]))
    hdr = " ".join(f"{'@'+str(k):>8s}" for k in KS)
    print(f"\n{'methode':16s} {'cout':>6s} {'biais':>8s} {'RMSE':>7s} | "
          f"rappel de masse {hdr}")
    print("-" * 108)
    for m, d in rows:
        cells = " ".join(f"{100*np.mean(d[k]):7.2f}%" for k in KS)
        print(f"{m:16s} {costs[m]:6.2f} {np.mean(d['bias']):+8.2f} "
              f"{np.mean(d['rmse']):7.3f} |                 {cells}")
    return {m: dict(cost=costs[m], bias=float(np.mean(d["bias"])),
                    rmse=float(np.mean(d["rmse"])),
                    **{f"recall@{k}": float(np.mean(d[k])) for k in KS})
            for m, d in acc.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen8k")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--local", type=int, default=8)
    a = ap.parse_args()

    out = {}
    for rope in (True, False):
        acc, costs, conc, nf = evaluate(a.tag, Lb=a.block, local=a.local, use_rope=rope)
        lbl = "avec RoPE (drop-in reel)" if rope else "sans RoPE (cadre COBS)"
        out[lbl] = report(acc, costs, conc, nf,
                          f"Selection de blocs distants -- L={a.block}, "
                          f"fenetre locale exclue={a.local} blocs -- {lbl}")
        print()
    (RES / f"selection_distant_L{a.block}.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES / f'selection_distant_L{a.block}.json'}")
