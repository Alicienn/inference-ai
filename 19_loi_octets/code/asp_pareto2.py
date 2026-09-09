"""
FRONTIERE DE PARETO OCTETS <-> RAPPEL DE LA SELECTION A DEUX PASSES.

ASP (papier 17_papier/asp.tex) : passe 1 lit un resume grossier de TOUS les blocs,
passe 2 lit un resume fin des m1 survivants. Le papier utilise des coresets dans les
deux passes. On teste ici si une passe 2 QUANTIFIEE (famille a taux exponentiel)
domine, et on mesure le cout total en octets par requete :

    octets = n * S1 + m1 * S2        (n = N/L blocs, m = blocs lus)

Baselines plates : un seul resume pour tous les blocs (octets = n * S).
Reference : passe 2 exacte (borne ce que la passe 1 peut couter en rappel).
"""
import numpy as np, json, pathlib, argparse, collections, sys, time
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\12_poc\code")
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code")
import summaries as S
from eval_selection import load
import frontiere2 as F2

OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
M = 8


def scores_for(fam, Ks, Q, s, Lb, D):
    """Retourne (J, n) scores et le cout en octets du resume."""
    n = Ks.shape[0]
    if fam == "exact":
        return F2.lse(s * np.einsum("nld,jd->nlj", Ks, Q), 1).T, Lb * D * 2
    F = F2.build(Ks, Lb, D)
    if fam not in F:
        raise KeyError(fam)
    kind, d, nb = F[fam]
    return F2.score_family(kind, d, Q, s).T, nb


def run(tag, Lb=64, nq=24, local=8, use_rope=True, seed=0, hstride=1, max_cfg=None,
        p1=("coreset r=1", "coreset r=2", "mean"), p2=("coreset r=4", "coreset r=8",
                                                       "coreset r=16", "quant 2b",
                                                       "quant 4b", "quant 6b", "exact"),
        mult=(2, 4, 8, 16), flat=("coreset r=1", "coreset r=2", "coreset r=4",
                                  "coreset r=8", "coreset r=16", "coreset r=32",
                                  "quant 1b", "quant 2b", "quant 3b", "quant 4b",
                                  "quant 6b", "quant 8b")):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")
    asp = collections.defaultdict(lambda: {"b": [], "r": [], "n": []})
    flt = collections.defaultdict(lambda: {"b": [], "r": [], "n": []})
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
            n = len(cand)
            cache = {}
            for h in range(kv * g, (kv + 1) * g, hstride):
                Q = Q_all[qpos, h, :].astype(np.float32)
                for fam in set(list(p1) + list(p2) + list(flat)):
                    if fam not in cache:
                        cache[fam] = scores_for(fam, Ks, Q, s, Lb, D)
                Tm = cache["exact"][0]
                mass = np.exp(Tm - Tm.max(1, keepdims=True))
                order = np.argsort(-Tm, 1)
                ora = np.take_along_axis(mass, order[:, :M], 1).sum(1)
                # --- baselines plates ---
                for fam in flat:
                    sc, nb = cache[fam]
                    om = np.argsort(-sc, 1)
                    got = np.take_along_axis(mass, om[:, :M], 1).sum(1)
                    flt[fam]["b"].append(n * nb); flt[fam]["r"].append(float(np.mean(got / ora)))
                    flt[fam]["n"].append(n)
                # --- deux passes ---
                for f1 in p1:
                    sc1, b1 = cache[f1]
                    for m1 in [M * k for k in mult]:
                        if m1 >= n:
                            continue
                        cand2 = np.argsort(-sc1, 1)[:, :m1]
                        for f2 in p2:
                            sc2, b2 = cache[f2]
                            sc2c = np.take_along_axis(sc2, cand2, 1)
                            ord2 = np.argsort(-sc2c, 1)[:, :M]
                            sel = np.take_along_axis(cand2, ord2, 1)
                            got = np.take_along_axis(mass, sel, 1).sum(1)
                            key = f"{f1} -> {f2} (m1={m1//M}m)"
                            asp[key]["b"].append(n * b1 + m1 * b2)
                            asp[key]["r"].append(float(np.mean(got / ora)))
                            asp[key]["n"].append(n)
    out = {"asp": {}, "flat": {}, "n_blocs_moyen": float(np.mean(flt[flat[0]]["n"]))}
    for k, v in asp.items():
        out["asp"][k] = dict(octets=float(np.mean(v["b"])), rappel=float(np.mean(v["r"])),
                             s1=float(np.mean(v["b"]) / np.mean(v["n"]) if v["n"] else 0))
    for k, v in flt.items():
        out["flat"][k] = dict(octets=float(np.mean(v["b"])), rappel=float(np.mean(v["r"])))
    return out


def pareto(items, tol=0.002):
    """items : liste de (octets, rappel, nom). Retourne l'enveloppe non dominee."""
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
            print("=" * 96)
            print(f"PARETO ASP  {lbl}  L={a.block}  n moyen={res['n_blocs_moyen']:.0f}  "
                  f"({time.time()-t0:.0f}s)")
            print("=" * 96)
            items = [(v["octets"], v["rappel"], f"PLAT {k}") for k, v in res["flat"].items()]
            items += [(v["octets"], v["rappel"], k) for k, v in res["asp"].items()]
            for b, r, nm in pareto(items):
                print(f"  {b:11.0f} o  {100*r:6.2f}%   {nm}")
            print("-" * 96)
            print("  comparaison a rappel egal (meilleur plat vs meilleur deux-passes) :")
            for target in (0.90, 0.95, 0.97, 0.99, 0.995):
                bp = min([b for b, r, _ in items if r >= target and "PLAT" in _], default=None)
                ba = min([b for b, r, _ in items if r >= target and "PLAT" not in _], default=None)
                if bp and ba:
                    print(f"    rappel >= {100*target:.1f}% : plat {bp:11.0f} o | "
                          f"deux-passes {ba:11.0f} o | gain {bp/ba:5.2f}x")
            fp = OUT / a.out
            prev = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
            prev[lbl] = res
            fp.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print("->", OUT / a.out, f"({time.time()-t0:.0f}s)")
