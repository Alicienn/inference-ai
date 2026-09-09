"""
ASP — mesure honnête du gain, par interpolation du front plat.

Méthode. Pour chaque configuration ASP, on cherche le nombre d'octets qu'il faudrait
à un schéma PLAT (résolution uniforme, passe unique) pour atteindre EXACTEMENT le
même rappel, en interpolant la courbe plate rappel(octets). Le rapport donne le gain
réel, sans dépendre du choix arbitraire d'un seuil de rappel.

Les points ASP dont le rappel dépasse le meilleur point plat mesuré sont exclus du
calcul (pas d'extrapolation).
"""
import numpy as np, json, pathlib, argparse
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def run(tag, Lb=64, nq=48, local=8, use_rope=True, seed=0, topk=8,
        rs=(1, 2, 4, 8, 16, 32), ms=(10, 12, 16, 20, 24, 32, 40, 48)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

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
            cand = np.arange(1, hi); n = len(cand)
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
            Sc = {r: np.stack([S.score(sums[r][b], Q, s) for b in range(n)], 1)
                  for r in rs}
            for r in rs:
                om = np.argsort(-Sc[r], 1)[:, :topk]
                got = np.take_along_axis(mass, om, 1).sum(1)
                flat[r]["rec"].append(float(np.mean(got / np.maximum(ora, 1e-30))))
                flat[r]["byt"].append(cost[r] * n)
            keep = np.argsort(-Sc[1], 1)
            for m in ms:
                if m > n:
                    continue
                sv = keep[:, :m]
                for r in rs:
                    if r == 1:
                        continue
                    fine = np.take_along_axis(Sc[r], sv, 1)
                    sel = np.take_along_axis(sv, np.argsort(-fine, 1)[:, :topk], 1)
                    got = np.take_along_axis(mass, sel, 1).sum(1)
                    asp[(m, r)]["rec"].append(float(np.mean(got / np.maximum(ora, 1e-30))))
                    asp[(m, r)]["byt"].append(cost[1] * n + cost[r] * m)
    return flat, asp


def report(flat, asp, lbl):
    fb = np.array([np.mean(d["byt"]) for r, d in sorted(flat.items())])
    fr = np.array([np.mean(d["rec"]) for r, d in sorted(flat.items())])
    o = np.argsort(fb); fb, fr = fb[o], fr[o]
    print("=" * 104)
    print(f"ASP vs schéma plat — gain en octets à rappel ÉGAL   ({lbl})")
    print("=" * 104)
    print("  Front plat de référence (résolution uniforme, passe unique) :")
    for r, b, rr in zip(sorted(flat), fb, fr):
        print(f"    coreset r={r:2d} : {b:8.1f} vec  ->  {100*rr:6.2f}%")
    print()
    print(f"  {'config ASP':16s} {'octets ASP':>11s} {'rappel':>8s} "
          f"{'octets plat équiv.':>19s} {'GAIN':>8s}")
    print("-" * 104)
    rows, gains = [], []
    for (m, r), d in sorted(asp.items()):
        if not d["rec"]:
            continue
        b, rc = np.mean(d["byt"]), np.mean(d["rec"])
        if rc > fr.max() or rc < fr.min():
            continue                                    # pas d'extrapolation
        beq = float(np.interp(rc, fr, fb))
        rows.append((m, r, b, rc, beq, beq / b))
    rows.sort(key=lambda z: -z[5])
    for m, r, b, rc, beq, gnn in rows[:14]:
        gains.append(gnn)
        print(f"  m={m:2d}, r={r:2d}      {b:11.1f} {100*rc:7.2f}% "
              f"{beq:19.1f} {gnn:7.2f}x")
    if rows:
        allg = [z[5] for z in rows]
        print(f"\n  gain médian sur toutes les configurations ASP valides : "
              f"{np.median(allg):.2f}x   (min {min(allg):.2f}x, max {max(allg):.2f}x)")
        print(f"  configurations où ASP perd (gain < 1) : "
              f"{sum(1 for x in allg if x < 1)}/{len(allg)}")
    return rows


if __name__ == "__main__":
    out = {}
    for tag in ("qwen8k", "smol8k"):
        for rope in (True, False):
            flat, asp = run(tag, use_rope=rope)
            lbl = f"{tag}  {'avec' if rope else 'sans'} RoPE"
            rows = report(flat, asp, lbl)
            out[lbl] = dict(
                flat={str(r): [float(np.mean(d["byt"])), float(np.mean(d["rec"]))]
                      for r, d in flat.items()},
                asp=[[m, r, float(b), float(rc), float(beq), float(gn)]
                     for m, r, b, rc, beq, gn in rows])
            print()
    (RES / "asp_pareto.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'asp_pareto.json'}")
