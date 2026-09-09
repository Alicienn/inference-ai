"""
LOI OCTETS-PRECISION DES RESUMES DE BLOCS  (frontiere debit-distorsion)

Question. Quelle est la loi sigma(b) entre l'erreur de score d'un resume de bloc
et son budget b en octets ? L'exposant alpha de sigma ~ b^-alpha gouverne tout :
le cout de la moindre amelioration de rappel, et la faisabilite de l'ASP.

Familles couvertes (toutes evaluees au meme protocole, memes blocs, memes requetes):
  mean        mu + effectif                          (premier ordre / NSA)
  cobs r      mu + covariance rang r                 (moments / COBS)
  coreset r   r centroides + effectifs (k-centre)    (support)
  gmm r x rk  melange gaussien                       (famille unifiee)
  quant k     TOUTES les cles quantifiees a k bits   (NOUVEAU : erreur ~ mode commun)
  sub r       r cles stockees exactement (sous-echantillon)  (NOUVEAU)
  top r       les r cles de plus grande norme        (NOUVEAU)

Metriques. sigma_disc = ecart-type ENTRE BLOCS de l'erreur, par requete
(c'est la grandeur qui predit le rappel, cf. law_selection.json) ; sigma_perp =
composante orthogonale au signal vrai ; rappel de masse @4/@8/@16.

Diagnostic geometrique. Pour chaque bloc, rayons gloutons d_1 >= d_2 >= ... du
k-centre. Theoreme utilise : rho_opt(r) >= d_r / 2 (les r+1 centres gloutons sont
2 a 2 separes d'au moins d_r), donc le coreset est a un facteur <= 2 de l'optimum
de sa famille. L'exposant de d_r mesure la dimension de recouvrement du nuage.
"""
import numpy as np, json, pathlib, argparse, collections, sys, time
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\12_poc\code")
import summaries as S
from eval_selection import load

OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
KS = (4, 8, 16)


def build_families(L, D):
    """Retourne {nom: (fonction K -> (summ, octets), score(K_dequant, Q, s))}"""
    F = {}

    def add(name, fn):
        F[name] = fn

    add("mean", lambda K: (S.s_mean(K)[0], (D + 1) * 2))
    for r in (1, 2, 4, 8, 16, 32, 64):
        if r <= L:
            add(f"coreset r={r}", (lambda r: lambda K: (S.s_coreset(K, r)[0],
                                                          r * (D + 1) * 2))(r))
    for r in (1, 2, 4, 8, 16):
        if r <= L:
            add(f"cobs r={r}", (lambda r: lambda K: (S.s_cobs(K, r)[0],
                                                     (D + r * D + r) * 2))(r))
    for (r, rk) in ((2, 1), (4, 1), (8, 1)):
        add(f"gmm {r}x{rk}", (lambda r, rk: lambda K: (S.s_gmm(K, r, rk)[0],
                                                       (r * D + r + r * rk * D + r * rk) * 2))(r, rk))
    for k in (2, 3, 4, 6, 8):
        add(f"quant {k}b", (lambda k: lambda K: (_quant(K, k), L * D * k / 8.0 + 4))(k))
    for r in (4, 8, 16, 32):
        if r <= L:
            add(f"sub r={r}", (lambda r: lambda K: (_sub(K, r), r * D * 2 + 2))(r))
            add(f"top r={r}", (lambda r: lambda K: (_top(K, r), r * D * 2 + 2))(r))
    return F


def _quant(K, k):
    """Quantification scalaire uniforme par bloc, k bits par dimension."""
    lo = K.min(0); hi = K.max(0)
    rng = np.maximum(hi - lo, 1e-12)
    q = np.round((K - lo) / rng * (2 ** k - 1))
    Kq = lo + q / (2 ** k - 1) * rng
    return dict(kind="quant", K=Kq)


def _sub(K, r):
    idx = np.linspace(0, K.shape[0] - 1, r).astype(int)
    return dict(kind="sub", K=K[idx], ratio=K.shape[0] / r)


def _top(K, r):
    idx = np.argsort(-np.linalg.norm(K, axis=1))[:r]
    return dict(kind="sub", K=K[idx], ratio=K.shape[0] / r)


def score_any(summ, Q, s):
    k = summ["kind"]
    if k in ("quant", "sub"):
        return S.true_logmass(summ["K"], Q, s) + (np.log(summ["ratio"]) if k == "sub" else 0.0)
    return S.score(summ, Q, s)


def greedy_radii(K, rmax):
    """d_r = rayon glouton du k-centre apres r centres (d_1 >= d_2 >= ...)."""
    L = K.shape[0]
    c = int(np.argmax(np.linalg.norm(K - K.mean(0), axis=1)))
    d = np.linalg.norm(K - K[c], axis=1)
    out = [float(d.max())]
    for _ in range(rmax - 1):
        i = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(K - K[i], axis=1))
        out.append(float(d.max()))
    return np.array(out)


def run(tag, Lb=64, nq=48, local=8, use_rope=True, seed=0, layer_stride=1):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")
    F = build_families(Lb, D)

    acc = {m: {"err": [], "sig": [], "sperp": [], "rmse": [], "bias": [],
               **{k: [] for k in KS}} for m in F}
    byts = {}
    radii = collections.defaultdict(list)
    rho_cs = collections.defaultdict(list)

    for (doc, layer), p in sorted(packs.items())[::layer_stride]:
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < max(KS) + 4:
                continue
            cand = np.arange(1, hi)
            sums = {m: [] for m in F}
            for b in cand:
                Kb = K[b * Lb:(b + 1) * Lb]
                for m, fn in F.items():
                    su, nb = fn(Kb)
                    sums[m].append(su); byts[m] = nb
                if len(radii) < 10 ** 9:
                    radii["all"].append(greedy_radii(Kb, min(Lb, 32)))
            for h in range(kv * g, (kv + 1) * g):
                Q = Q_all[qpos, h, :].astype(np.float32)
                Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s) for b in cand], 1)
                mass = np.exp(Tm - Tm.max(1, keepdims=True))
                order = np.argsort(-Tm, 1)
                sig_c = Tm - Tm.mean(1, keepdims=True)
                for m in F:
                    Sc = np.stack([score_any(sums[m][i], Q, s) for i in range(len(cand))], 1)
                    e = Sc - Tm
                    eps = e - e.mean(1, keepdims=True)
                    acc[m]["err"].append(e)
                    acc[m]["sig"].append(eps.std(1).mean())
                    acc[m]["rmse"].append(float(np.sqrt((e ** 2).mean())))
                    acc[m]["bias"].append(float(e.mean()))
                    den = (sig_c ** 2).sum(1, keepdims=True)
                    a = (eps * sig_c).sum(1, keepdims=True) / np.maximum(den, 1e-30)
                    perp = eps - a * sig_c
                    acc[m]["sperp"].append(float(perp.std(1).mean()))
                    om = np.argsort(-Sc, 1)
                    for k in KS:
                        got = np.take_along_axis(mass, om[:, :k], 1).sum(1)
                        ora = np.take_along_axis(mass, order[:, :k], 1).sum(1)
                        acc[m][k].append(float(np.mean(got / np.maximum(ora, 1e-30))))

    out = {}
    for m in F:
        d = acc[m]
        out[m] = dict(bytes=float(byts[m]), rmse=float(np.mean(d["rmse"])),
                      bias=float(np.mean(d["bias"])),
                      sigma_disc=float(np.mean(d["sig"])),
                      sigma_perp=float(np.mean(d["sperp"])),
                      **{f"r@{k}": float(np.mean(d[k])) for k in KS})
    return out, radii


def fit_alpha(pts):
    pts = sorted([(b, s) for b, s in pts if s > 0])
    if len(pts) < 2:
        return None
    x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts])
    a, b = np.polyfit(x, y, 1)
    return float(np.exp(b)), float(-a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="qwen8k,smol8k")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--nq", type=int, default=48)
    a = ap.parse_args()
    allout = {}
    t0 = time.time()
    for tag in a.tags.split(","):
        for rope in (True, False):
            res, radii = run(tag, Lb=a.block, nq=a.nq, use_rope=rope)
            lbl = f"{tag}|{'RoPE' if rope else 'NoPE'}"
            allout[lbl] = res
            print("=" * 104)
            print(f"FRONTIERE OCTETS-PRECISION  {lbl}  (L={a.block}, {time.time()-t0:.0f}s)")
            print("=" * 104)
            print(f"{'famille':16s} {'octets':>8s} {'RMSE':>7s} {'sigma_disc':>10s} "
                  f"{'sigma_perp':>10s} {'r@4':>7s} {'r@8':>7s} {'r@16':>7s}")
            print("-" * 104)
            for m, v in sorted(res.items(), key=lambda kv: kv[1]["bytes"]):
                print(f"{m:16s} {v['bytes']:8.0f} {v['rmse']:7.3f} {v['sigma_disc']:10.4f} "
                      f"{v['sigma_perp']:10.4f} {100*v['r@4']:6.2f}% {100*v['r@8']:6.2f}% "
                      f"{100*v['r@16']:6.2f}%")
            # exposants par famille
            print("-" * 104)
            for fam, pref in (("coreset", "coreset r="), ("cobs", "cobs r="),
                              ("quant", "quant "), ("sub", "sub r="), ("top", "top r=")):
                pts = [(v["bytes"], v["sigma_disc"]) for m, v in res.items() if m.startswith(pref)]
                f = fit_alpha(pts)
                if f:
                    print(f"  {fam:8s} sigma_disc = {f[0]:8.4f} * b^(-{f[1]:.3f})"
                          f"   -> doubler b divise sigma par {2**f[1]:.3f}"
                          f"   -> pour diviser sigma par 2 : x{2**(1/f[1]):.1f} octets")
            print()
    (OUT / "frontiere.json").write_text(json.dumps(allout, indent=2), encoding="utf-8")
    print("->", OUT / "frontiere.json", f"({time.time()-t0:.0f}s)")
