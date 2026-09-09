"""PARETO OCTETS<->RAPPEL DE LA SELECTION A DEUX PASSES (v3, corrige).

octets = n*S1 + m1*S2 ; baselines plates : n*S. Passe 2 exacte = borne du rappel
atteignable par la passe 1.
"""
import numpy as np, json, pathlib, argparse, collections, sys, time
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\12_poc\code")
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code")
from eval_selection import load
import frontiere2 as F2

OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
M = 8
P1 = ("coreset r=1", "coreset r=2", "quantG 2b")
P2 = ("coreset r=4", "coreset r=8", "coreset r=16", "quant 2b", "quant 4b", "quant 6b", "exact")
FLAT = ("coreset r=1", "coreset r=2", "coreset r=4", "coreset r=8", "coreset r=16",
        "coreset r=32", "quant 1b", "quant 2b", "quant 3b", "quant 4b", "quant 6b")
NEED = sorted(set(P1 + P2 + FLAT) - {"exact"})


def run(tag, Lb=64, nq=24, local=8, use_rope=True, seed=0, hstride=2, max_cfg=None,
        mult=(2, 4, 8, 16)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")
    asp = collections.defaultdict(lambda: {"b": [], "r": []})
    flt = collections.defaultdict(lambda: {"b": [], "r": []})
    ns = []
    cfgs = sorted(packs.items())
    if max_cfg:
        cfgs = cfgs[:max_cfg]
    for (doc, layer), p in cfgs:
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < M + 4:
                continue
            cand = np.arange(1, hi)
            Ks = np.stack([K[b * Lb:(b + 1) * Lb] for b in cand])
            n = len(cand); ns.append(n)
            FAM = F2.build(Ks, Lb, D)          # construit UNE fois par (cfg, kv)
            for h in range(kv * g, (kv + 1) * g, hstride):
                Q = Q_all[qpos, h, :].astype(np.float32)
                sc = {}
                for fam in NEED:
                    kind, d, nb = FAM[fam]
                    sc[fam] = (F2.score_family(kind, d, Q, s).T, nb)
                Tm, b_ex = F2.lse(s * np.einsum("nld,jd->nlj", Ks, Q), 1).T, Lb * D * 2
                sc["exact"] = (Tm, b_ex)
                mass = np.exp(Tm - Tm.max(1, keepdims=True))
                order = np.argsort(-Tm, 1)
                ora = np.take_along_axis(mass, order[:, :M], 1).sum(1)
                for fam in FLAT:
                    s0, nb = sc[fam]
                    om = np.argsort(-s0, 1)
                    got = np.take_along_axis(mass, om[:, :M], 1).sum(1)
                    flt[fam]["b"].append(n * nb); flt[fam]["r"].append(float(np.mean(got / ora)))
                for f1 in P1:
                    s1, b1 = sc[f1]
                    a1 = np.argsort(-s1, 1)
                    for m1 in [M * k for k in mult]:
                        if m1 >= n:
                            continue
                        cand2 = a1[:, :m1]
                        for f2 in P2:
                            s2, b2 = sc[f2]
                            s2c = np.take_along_axis(s2, cand2, 1)
                            ord2 = np.argsort(-s2c, 1)[:, :M]
                            sel = np.take_along_axis(cand2, ord2, 1)
                            got = np.take_along_axis(mass, sel, 1).sum(1)
                            key = f"{f1} -> {f2} m1={m1//M}m"
                            asp[key]["b"].append(n * b1 + m1 * b2)
                            asp[key]["r"].append(float(np.mean(got / ora)))
    out = {"asp": {k: dict(octets=float(np.mean(v["b"])), rappel=float(np.mean(v["r"])))
                   for k, v in asp.items()},
           "flat": {k: dict(octets=float(np.mean(v["b"])), rappel=float(np.mean(v["r"])))
                    for k, v in flt.items()},
           "n_blocs_moyen": float(np.mean(ns))}
    return out


def pareto(items, tol=0.0015):
    items = sorted(items)
    best = -1; keep = []
    for b, r, nm in items:
        if r > best + tol:
            keep.append((b, r, nm)); best = r
    return keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="qwen8k")
    ap.add_argument("--rope", default="true")
    ap.add_argument("--nq", type=int, default=24)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--maxcfg", type=int, default=None)
    ap.add_argument("--out", default="asp_pareto.json")
    a = ap.parse_args()
    t0 = time.time()
    for tag in a.tags.split(","):
        for rope in ([True, False] if a.rope == "both" else [a.rope == "true"]):
            res = run(tag, Lb=a.block, nq=a.nq, use_rope=rope, hstride=a.heads,
                      max_cfg=a.maxcfg)
            lbl = f"{tag}|{'RoPE' if rope else 'NoPE'}"
            print("=" * 92)
            print(f"PARETO ASP  {lbl}  L={a.block}  n={res['n_blocs_moyen']:.0f}  "
                  f"({time.time()-t0:.0f}s)")
            print("=" * 92)
            items = [(v["octets"], v["rappel"], "PLAT " + k) for k, v in res["flat"].items()]
            items += [(v["octets"], v["rappel"], k) for k, v in res["asp"].items()]
            for b, r, nm in pareto(items):
                print(f"  {b:11.0f} o  {100*r:6.2f}%   {nm}")
            print("-" * 92)
            for target in (0.90, 0.95, 0.97, 0.99, 0.995, 0.999):
                bp = min([b for b, r, nm in items if r >= target and nm.startswith("PLAT")],
                         default=None)
                ba = min([b for b, r, nm in items if r >= target and not nm.startswith("PLAT")],
                         default=None)
                if bp and ba:
                    print(f"    rappel >= {100*target:5.1f}% : plat {bp:10.0f} o | "
                          f"2 passes {ba:10.0f} o | gain {bp/ba:5.2f}x")
            fp = OUT / a.out
            prev = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
            prev[lbl] = res
            fp.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print("->", OUT / a.out, f"({time.time()-t0:.0f}s)")
