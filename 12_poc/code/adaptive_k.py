"""
CONCEPT 4 -- budget ADAPTATIF, consequence directe de la structure d'ex aequo.

Le diagnostic diag_cert.py a etabli que l'ecart de log-masse entre le k-ieme et le
(k+1)-ieme bloc vaut ~0.07 nat en mediane : les blocs autour de la coupure sont
quasi indiscernables. Mais cette mediane cache une variance enorme : pour
certaines requetes un seul bloc concentre la masse, pour d'autres elle est etalee
sur des dizaines de blocs.

Un top-k FIXE est donc doublement inadapte : trop genereux quand la masse est
concentree, trop avare quand elle est etalee. Or le coreset fournit une ESTIMATION
de la masse de chaque bloc : on peut donc choisir k par requete de sorte que la
masse estimee cumulee atteigne une cible tau.

On compare a BUDGET MOYEN EGAL : k adaptatif de moyenne kbar contre k fixe = kbar.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0, r=4,
        taus=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    rows = {t: dict(k=[], rec=[]) for t in taus}
    fixed = {}
    spread = []

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 24:
                continue
            cand = np.arange(1, hi)
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            sums = [S.s_coreset(K[b * Lb:(b + 1) * Lb], r) [0] for b in cand]
            Sc = np.stack([S.score(su, Q, s) for su in sums], 1)
            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                           for b in cand], 1)
            mass = np.exp(Tm - Tm.max(1, keepdims=True))
            mass = mass / mass.sum(1, keepdims=True)          # part de masse vraie
            order = np.argsort(-Sc, 1)
            true_order = np.argsort(-Tm, 1)
            # nombre de blocs vraiment necessaires pour 90% de la masse
            cs_true = np.cumsum(np.take_along_axis(mass, true_order, 1), 1)
            spread.append(float(np.mean((cs_true < 0.9).sum(1) + 1)))

            est = np.exp(Sc - Sc.max(1, keepdims=True))
            est = est / est.sum(1, keepdims=True)
            est_sorted = np.take_along_axis(est, order, 1)
            cum = np.cumsum(est_sorted, 1)
            got_sorted = np.take_along_axis(mass, order, 1)
            got_cum = np.cumsum(got_sorted, 1)

            for t in taus:
                kk_ = (cum < t).sum(1) + 1
                kk_ = np.minimum(kk_, len(cand))
                rows[t]["k"].append(float(kk_.mean()))
                rec = got_cum[np.arange(len(Q)), kk_ - 1]
                rows[t]["rec"].append(float(rec.mean()))
            for kf in (2, 4, 6, 8, 12, 16, 24, 32):
                if kf <= len(cand):
                    fixed.setdefault(kf, []).append(
                        float(got_cum[:, kf - 1].mean()))
    return rows, fixed, float(np.mean(spread))


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        rows, fixed, spread = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        print("=" * 92)
        print(f"CONCEPT 4 -- budget adaptatif vs top-k fixe  --  {lbl}  (coreset r=4, L=64)")
        print("=" * 92)
        print(f"  blocs necessaires pour 90% de la masse (oracle) : {spread:.1f} en moyenne")
        print(f"\n  {'cible tau':>10s} {'k moyen':>9s} {'masse captee':>14s}   |   "
              f"{'k fixe':>7s} {'masse captee':>14s}")
        print("-" * 92)
        fx = sorted(fixed.items())
        adapt = [(t, np.mean(d['k']), np.mean(d['rec'])) for t, d in rows.items()]
        for i, (t, km, rc) in enumerate(adapt):
            # k fixe le plus proche du k moyen adaptatif
            kf, rf = min(fx, key=lambda z: abs(z[0] - km))
            print(f"  {t:10.2f} {km:9.2f} {100*rc:13.2f}%   |   {kf:7d} "
                  f"{100*np.mean(rf):13.2f}%   {'<-- adaptatif gagne' if rc>np.mean(rf) else ''}")
        print(f"\n  reference, k fixe :")
        for kf, v in fx:
            print(f"    k={kf:3d} -> {100*np.mean(v):6.2f}% de la masse")
        out[lbl] = dict(spread90=spread,
                        adaptive=[(float(t), float(k), float(r)) for t, k, r in adapt],
                        fixed={str(k): float(np.mean(v)) for k, v in fx})
        print()
    (RES / "adaptive_k.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'adaptive_k.json'}")
