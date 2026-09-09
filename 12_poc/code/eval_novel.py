"""
Evaluation des deux concepts nouveaux :
  (1) EQS  -- resume par quadrature exponentielle (poids ajustes)
  (2) selection CERTIFIEE via la borne bilaterale
"""
import numpy as np, json, pathlib
import summaries as S, novel as N
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"
KS = (1, 2, 4, 8)


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    meth = {}
    for r in (2, 4, 8):
        meth[f"coreset r={r}"] = ("plain", r)
        meth[f"EQS r={r}"] = ("eqs", r)
    meth["cobs r=4"] = ("cobs", 4)
    meth["cobs r=8"] = ("cobs", 8)
    meth["mean"] = ("mean", 0)

    acc = {m: {k: [] for k in KS} for m in meth}
    for m in meth:
        acc[m]["rmse"] = []
    cost = {}
    # certificats : pour chaque r, taille de l'ensemble ambigu et couverture
    cert = {r: dict(amb=[], cov=[], amb_aniso=[], nb=[]) for r in (2, 4, 8)}

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 12:
                continue
            cand = np.arange(1, hi)
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            sig = N.calib_sigma(Q)

            # construction des resumes
            sums, labs = {m: [] for m in meth}, {}
            for r in (2, 4, 8):
                labs[r] = []
            for b in cand:
                Kb = K[b * Lb:(b + 1) * Lb]
                for m, (kind, r) in meth.items():
                    if kind == "plain":
                        su, c = S.s_coreset(Kb, r)
                    elif kind == "eqs":
                        su, c = N.s_eqs(Kb, r, sigma=sig)
                    elif kind == "cobs":
                        su, c = S.s_cobs(Kb, r)
                    else:
                        su, c = S.s_mean(Kb)
                    sums[m].append(su); cost[m] = c
                for r in (2, 4, 8):
                    labs[r].append(S.kcenter(Kb, r))

            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                           for b in cand], 1)
            mass = np.exp(Tm - Tm.max(1, keepdims=True))
            order = np.argsort(-Tm, 1)

            for m in meth:
                Sc = np.stack([S.score(sums[m][i], Q, s) for i in range(len(cand))], 1)
                acc[m]["rmse"].append(float(np.sqrt(((Sc - Tm) ** 2).mean())))
                om = np.argsort(-Sc, 1)
                for k in KS:
                    got = np.take_along_axis(mass, om[:, :k], 1).sum(1)
                    ora = np.take_along_axis(mass, order[:, :k], 1).sum(1)
                    acc[m][k].append(float(np.mean(got / np.maximum(ora, 1e-30))))

            # --- certificats
            Qn = np.linalg.norm(Q, axis=1)
            for r in (2, 4, 8):
                Sc = np.stack([S.score(sums[f"coreset r={r}"][i], Q, s)
                               for i in range(len(cand))], 1)
                rho = np.array([sums[f"coreset r={r}"][i]["rho"] for i in range(len(cand))])
                amb, tau = N.certified_select(Sc, rho, Qn, s, 8)
                cert[r]["amb"].append(float(amb.sum(1).mean()))
                cert[r]["nb"].append(len(cand))
                # couverture : le vrai top-8 est-il inclus dans l'ensemble ambigu ?
                top = order[:, :8]
                covered = np.take_along_axis(amb, top, 1).all(1)
                cert[r]["cov"].append(float(covered.mean()))
                # borne anisotrope
                da = np.stack([N.anisotropic_delta(K[b * Lb:(b + 1) * Lb], labs[r][i], Q, s)
                               for i, b in enumerate(cand)], 1)
                Lo, Up = Sc - da, Sc + da
                tau2 = np.sort(Lo, 1)[:, -8][:, None]
                cert[r]["amb_aniso"].append(float((Up >= tau2).sum(1).mean()))
    return acc, cost, cert


def report(acc, cost, cert, lbl):
    print("=" * 100)
    print(f"CONCEPT 1 -- EQS : quadrature a poids ajustes   ({lbl})")
    print("=" * 100)
    rows = sorted(acc.items(), key=lambda kv: (cost[kv[0]], kv[0]))
    hdr = " ".join(f"{'@'+str(k):>8s}" for k in KS)
    print(f"{'methode':16s} {'cout':>6s} {'RMSE':>7s} |  rappel de masse {hdr}")
    print("-" * 100)
    for m, d in rows:
        cells = " ".join(f"{100*np.mean(d[k]):7.2f}%" for k in KS)
        print(f"{m:16s} {cost[m]:6.2f} {np.mean(d['rmse']):7.3f} |                  {cells}")

    print()
    print("=" * 100)
    print(f"CONCEPT 2 -- selection CERTIFIEE (top-8 garanti)   ({lbl})")
    print("=" * 100)
    print(f"{'r':>3s} {'blocs candidats':>16s} {'ambigus (iso)':>15s} "
          f"{'ambigus (aniso)':>17s} {'couverture':>12s}")
    print("-" * 100)
    for r, d in cert.items():
        nb = np.mean(d["nb"])
        print(f"{r:3d} {nb:16.1f} {np.mean(d['amb']):15.1f} "
              f"{np.mean(d['amb_aniso']):17.1f} {100*np.mean(d['cov']):11.2f}%")
    print("\n  'couverture' = frequence a laquelle le vrai top-8 est inclus dans")
    print("  l'ensemble ambigu. Le theoreme garantit 100% ; toute valeur < 100%")
    print("  signalerait un bug d'implementation.")


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        acc, cost, cert = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        report(acc, cost, cert, lbl)
        out[lbl] = dict(
            methods={m: dict(cost=cost[m], rmse=float(np.mean(d["rmse"])),
                             **{f"recall@{k}": float(np.mean(d[k])) for k in KS})
                     for m, d in acc.items()},
            cert={str(r): dict(nb=float(np.mean(d["nb"])), amb=float(np.mean(d["amb"])),
                               amb_aniso=float(np.mean(d["amb_aniso"])),
                               cov=float(np.mean(d["cov"]))) for r, d in cert.items()})
        print()
    (RES / "novel_results.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'novel_results.json'}")
