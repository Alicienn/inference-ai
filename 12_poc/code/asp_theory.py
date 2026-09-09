"""
ASP — validation théorique et réplication.

RÈGLE DE CONCEPTION DÉRIVÉE
---------------------------
[DÉRIVATION] Le schéma à deux passes ne perd un vrai bloc du top-k que si le bruit de
la passe grossière le fait sortir du top-m. Notons l_(j) la j-ième plus grande
log-masse vraie d'une requête. Un bloc de rang vrai j <= k survit à la passe 1 dès que
son score bruité dépasse celui du bloc de rang m, ce qui demande à peu près

    l_(k) - l_(m)  >  c * sigma_1 * sqrt(2)          (c ~ 1 a 2)

le sqrt(2) venant de ce que DEUX scores bruités sont comparés (variance doublée).
D'où la règle : **choisir le plus petit m tel que l'écart de log-masse entre le rang k
et le rang m dépasse ~2 sigma_1.**

On mesure ici la fonction m -> l_(k) - l_(m) sur données réelles, on en déduit le m
prédit, et on le compare au m optimal observé sur le front de Pareto. Si les deux
coïncident, la règle est utilisable sans balayage empirique.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def run(tag, Lb=64, nq=48, local=8, use_rope=True, seed=0, topk=8,
        ms=(10, 12, 16, 20, 24, 32, 40, 48), rs=(1, 4, 16)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    gaps = {m: [] for m in ms}
    sig1, surv = [], {m: [] for m in ms}
    flat = {r: dict(rec=[], byt=[]) for r in rs}
    asp = {(m, r): dict(rec=[], byt=[]) for m in ms for r in rs if r > 1}

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 56:
                continue
            cand = np.arange(1, hi)
            n = len(cand)
            sums, cost = {}, {}
            for r in rs:
                sums[r] = []
                for b in cand:
                    su, c = S.s_coreset(K[b * Lb:(b + 1) * Lb], r)
                    sums[r].append(su); cost[r] = c
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                           for b in cand], 1)
            mass = np.exp(Tm - Tm.max(1, keepdims=True))
            order = np.argsort(-Tm, 1)
            ora = np.take_along_axis(mass, order[:, :topk], 1).sum(1)
            srt = np.sort(Tm, 1)[:, ::-1]

            for m in ms:
                if m < n:
                    gaps[m].append(float(np.mean(srt[:, topk - 1] - srt[:, m - 1])))

            Sc = {r: np.stack([S.score(sums[r][b], Q, s) for b in range(n)], 1)
                  for r in rs}
            e = Sc[1] - Tm
            eps = e - e.mean(1, keepdims=True)
            sig1.append(float(eps.std(1).mean()))

            keep = np.argsort(-Sc[1], 1)
            for m in ms:
                if m > n:
                    continue
                top = order[:, :topk]
                inm = np.array([[t in set(keep[i, :m]) for t in top[i]]
                                for i in range(len(Q))])
                surv[m].append(float(inm.mean()))
                for r in rs:
                    if r == 1:
                        continue
                    sv = keep[:, :m]
                    fine = np.take_along_axis(Sc[r], sv, 1)
                    sel = np.take_along_axis(sv, np.argsort(-fine, 1)[:, :topk], 1)
                    got = np.take_along_axis(mass, sel, 1).sum(1)
                    asp[(m, r)]["rec"].append(float(np.mean(got / np.maximum(ora, 1e-30))))
                    asp[(m, r)]["byt"].append(cost[1] * n + cost[r] * m)
            for r in rs:
                om = np.argsort(-Sc[r], 1)[:, :topk]
                got = np.take_along_axis(mass, om, 1).sum(1)
                flat[r]["rec"].append(float(np.mean(got / np.maximum(ora, 1e-30))))
                flat[r]["byt"].append(cost[r] * n)
    return gaps, np.mean(sig1), surv, flat, asp, ms, rs


if __name__ == "__main__":
    out = {}
    for tag in ("qwen8k", "smol8k"):
        for rope in (True,):
            gaps, s1, surv, flat, asp, ms, rs = run(tag, use_rope=rope)
            lbl = f"{tag} / avec RoPE"
            print("=" * 100)
            print(f"ASP — validation de la règle de conception   ({lbl})")
            print("=" * 100)
            print(f"  bruit de la passe grossière (coreset r=1) : sigma_1 = {s1:.3f} nat")
            print(f"  seuil visé : ecart l_(8) - l_(m) > 2*sigma_1 = {2*s1:.3f} nat\n")
            print(f"{'m':>5s} {'ecart l_(8)-l_(m)':>18s} {'>2sigma1 ?':>11s} "
                  f"{'top-8 survivant':>16s}")
            print("-" * 100)
            m_pred = None
            for m in ms:
                if not gaps[m]:
                    continue
                gp = np.mean(gaps[m]); ok = gp > 2 * s1
                if ok and m_pred is None:
                    m_pred = m
                print(f"{m:5d} {gp:18.3f} {('oui' if ok else 'non'):>11s} "
                      f"{100*np.mean(surv[m]):15.2f}%")
            print(f"\n  -> m prédit par la règle : {m_pred}")

            # m optimal observé : meilleur rappel par octet autour du coude
            best = None
            for (m, r), d in asp.items():
                if not d["rec"]:
                    continue
                eff = np.mean(d["rec"]) / np.mean(d["byt"])
                if best is None or eff > best[0]:
                    best = (eff, m, r, np.mean(d["rec"]), np.mean(d["byt"]))
            print(f"  -> m optimal observé (meilleur rappel/octet) : m={best[1]} "
                  f"(r={best[2]}, rappel {100*best[3]:.2f}%)")

            print(f"\n  Gain ASP vs plat, à rappel comparable :")
            for r in rs:
                if r == 1:
                    continue
                fr, fb = np.mean(flat[r]["rec"]), np.mean(flat[r]["byt"])
                cands = [(np.mean(d["byt"]), np.mean(d["rec"]), m, rr)
                         for (m, rr), d in asp.items()
                         if d["rec"] and np.mean(d["rec"]) >= fr]
                if cands:
                    b, rc, m, rr = min(cands, key=lambda z: z[0])
                    print(f"    plat r={r:2d} : {100*fr:5.2f}% pour {fb:7.1f} vec"
                          f"   |  ASP m={m},r={rr} : {100*rc:5.2f}% pour {b:7.1f} vec"
                          f"   -> **{fb/b:.2f}x**")
            out[lbl] = dict(sigma1=float(s1), m_pred=m_pred,
                            gaps={str(m): float(np.mean(gaps[m])) for m in ms if gaps[m]},
                            surv={str(m): float(np.mean(surv[m])) for m in ms if surv[m]})
            print()
    (RES / "asp_theory.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'asp_theory.json'}")
