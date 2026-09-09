"""
LOI OCTETS-PRECISION DES RESUMES DE BLOCS -- version vectorisee.

Familles (toutes au meme protocole : memes blocs, memes requetes, memes candidats) :
  mean, cobs r, coreset r, gmm r x rk, quant k (toutes les cles quantifiees k bits),
  sub r (sous-echantillon exact), top r (r cles de plus grande norme).

Sortie : sigma_disc(b), sigma_perp(b), rappel de masse @4/@8/@16, ajustement de
sigma ~ b^-alpha par famille, et rayons gloutons d_r (dimension de recouvrement).
"""
import numpy as np, json, pathlib, argparse, collections, sys, time
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\12_poc\code")
import summaries as S
from eval_selection import load

OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
KS = (4, 8, 16)


def lse(A, axis):
    M = A.max(axis=axis, keepdims=True)
    R = M + np.log(np.exp(A - M).sum(axis=axis, keepdims=True))
    return np.squeeze(R, axis=axis)


def kcenter_lab(Ks, r):
    """Ks : (n, L, D) -> labels (n, L). Gonzalez vectorise."""
    n, L, D = Ks.shape
    if r >= L:
        return np.tile(np.arange(L), (n, 1))
    c = np.argmax(((Ks - Ks.mean(1, keepdims=True)) ** 2).sum(-1), axis=1)
    C = np.take_along_axis(Ks, c[:, None, None], axis=1)          # (n,1,D)
    d = ((Ks - C) ** 2).sum(-1)                                    # (n,L)
    lab = np.zeros((n, L), dtype=int)
    for j in range(1, r):
        i = np.argmax(d, axis=1)
        Ci = np.take_along_axis(Ks, i[:, None, None], axis=1)
        d = np.minimum(d, ((Ks - Ci) ** 2).sum(-1))
    # affectation finale aux r centres
    idx = np.zeros((n, r), dtype=int)
    d = ((Ks - Ks.mean(1, keepdims=True)) ** 2).sum(-1)
    c0 = np.argmax(d, axis=1); idx[:, 0] = c0
    C = np.take_along_axis(Ks, c0[:, None, None], axis=1)
    dmin = ((Ks - C) ** 2).sum(-1)
    for j in range(1, r):
        i = np.argmax(dmin, axis=1); idx[:, j] = i
        Ci = np.take_along_axis(Ks, i[:, None, None], axis=1)
        dmin = np.minimum(dmin, ((Ks - Ci) ** 2).sum(-1))
    C = np.take_along_axis(Ks, idx[:, :, None].repeat(D, 2), axis=1)  # (n,r,D)
    dist = ((Ks[:, :, None, :] - C[:, None, :, :]) ** 2).sum(-1)      # (n,L,r)
    return dist.argmin(-1)


def build(Ks, L, D):
    """Retourne {nom: (kind, dict d'arrays sur l'axe n, octets)}"""
    n = Ks.shape[0]
    F = {}
    F["mean"] = ("vec", dict(mu=Ks.mean(1), cnt=np.full(n, L)), (D + 1) * 2)
    for r in (1, 2, 4, 8, 16, 32, 64):
        if r > L:
            continue
        lab = kcenter_lab(Ks, r)
        mus = np.zeros((n, r, D), np.float32); ns = np.zeros((n, r), np.float32)
        for j in range(r):
            m = (lab == j)[:, :, None]
            ns[:, j] = m.sum(1)[:, 0]
            mus[:, j, :] = (Ks * m).sum(1) / np.maximum(ns[:, j:j + 1], 1)
        F[f"coreset r={r}"] = ("coreset", dict(mu=mus, cnt=ns), r * (D + 1) * 2)
    for r in (1, 2, 4, 8, 16):
        if r > L:
            continue
        mu = Ks.mean(1); X = Ks - mu[:, None, :]
        # SVD par bloc
        U = np.zeros((n, r, D), np.float32); lam = np.zeros((n, r), np.float32)
        for i in range(n):
            _, sv, vt = np.linalg.svd(X[i], full_matrices=False)
            rr = min(r, vt.shape[0]); U[i, :rr] = vt[:rr]; lam[i, :rr] = sv[:rr] ** 2 / L
        F[f"cobs r={r}"] = ("cobs", dict(mu=mu, U=U, lam=lam), (D + r * D + r) * 2)
    for (r, rk) in ((2, 1), (4, 1), (8, 1)):
        if r > L:
            continue
        lab = kcenter_lab(Ks, r)
        mus = np.zeros((n, r, D), np.float32); ns = np.zeros((n, r), np.float32)
        U = np.zeros((n, r, rk, D), np.float32); lam = np.zeros((n, r, rk), np.float32)
        for j in range(r):
            m = (lab == j)
            cnt = m.sum(1).astype(np.float32); ns[:, j] = cnt
            Kc = np.where(m[:, :, None], Ks, 0.0)
            mus[:, j, :] = Kc.sum(1) / np.maximum(cnt[:, None], 1)
            for i in range(n):
                if cnt[i] < 2:
                    continue
                Xc = Ks[i][m[i]] - mus[i, j]
                _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
                rr = min(rk, vt.shape[0]); U[i, j, :rr] = vt[:rr]
                lam[i, j, :rr] = sv[:rr] ** 2 / cnt[i]
        F[f"gmm {r}x{rk}"] = ("gmm", dict(mu=mus, cnt=ns, U=U, lam=lam),
                              (r * D + r + r * rk * D + r * rk) * 2)
    for k in (1, 2, 3, 4, 6, 8):
        # (a) echelles GLOBALES par dimension (calculees une fois au prefill, comme
        #     dans les quantifieurs KV de production : cout amorti)
        lo = Ks.min((0, 1), keepdims=True); hi = Ks.max((0, 1), keepdims=True)
        rng = np.maximum(hi - lo, 1e-12)
        q = np.round((Ks - lo) / rng * (2 ** k - 1))
        F[f"quantG {k}b"] = ("keys", dict(K=lo + q / (2 ** k - 1) * rng, ratio=np.ones(n)),
                             L * D * k / 8.0 + 2 * D * 2 / max(n, 1))
        # (b) echelles PAR BLOC et par dimension : +2 scalaires fp16 par dimension
        lo = Ks.min(1, keepdims=True); hi = Ks.max(1, keepdims=True)
        rng = np.maximum(hi - lo, 1e-12)
        q = np.round((Ks - lo) / rng * (2 ** k - 1))
        F[f"quant {k}b"] = ("keys", dict(K=lo + q / (2 ** k - 1) * rng, ratio=np.ones(n)),
                            L * D * k / 8.0 + 2 * D * 2)
    for r in (4, 8, 16, 32):
        if r > L:
            continue
        idx = np.linspace(0, L - 1, r).astype(int)
        F[f"sub r={r}"] = ("keys", dict(K=Ks[:, idx], ratio=np.full(n, L / r)),
                           r * D * 2 + 2)
        top = np.argsort(-np.linalg.norm(Ks, axis=2), axis=1)[:, :r]
        F[f"top r={r}"] = ("keys", dict(K=np.take_along_axis(Ks, top[:, :, None], 1),
                                        ratio=np.full(n, L / r)), r * D * 2 + 2)
    return F


def score_family(kind, d, Q, s):
    """Q : (J, D) -> (n, J) scores."""
    if kind == "vec":
        return s * (d["mu"] @ Q.T) + np.log(d["cnt"])[:, None]
    if kind == "coreset":
        A = s * np.einsum("nd,jd->nj", d["mu"].reshape(-1, d["mu"].shape[-1]), Q)
        A = A.reshape(d["mu"].shape[0], d["mu"].shape[1], -1) + np.log(d["cnt"])[:, :, None]
        return lse(A, 1)
    if kind == "cobs":
        lin = s * (d["mu"] @ Q.T)
        proj = s * np.einsum("nrd,jd->nrj", d["U"], Q)
        return lin + 0.5 * (proj ** 2 * d["lam"][:, :, None]).sum(1)
    if kind == "gmm":
        lin = s * np.einsum("nrd,jd->nrj", d["mu"], Q)
        proj = s * np.einsum("nrkd,jd->nrkj", d["U"], Q)
        quad = 0.5 * (proj ** 2 * d["lam"][:, :, :, None]).sum(2)
        return lse(lin + quad + np.log(d["cnt"])[:, :, None], 1)
    if kind == "keys":
        A = s * np.einsum("nld,jd->nlj", d["K"], Q)
        return lse(A, 1) + np.log(d["ratio"])[:, None]
    raise ValueError(kind)


def greedy_radii(Ks, rmax):
    n = Ks.shape[0]
    c = np.argmax(((Ks - Ks.mean(1, keepdims=True)) ** 2).sum(-1), axis=1)
    C = np.take_along_axis(Ks, c[:, None, None], axis=1)
    d = np.sqrt(((Ks - C) ** 2).sum(-1))
    out = [d.max(1)]
    for _ in range(rmax - 1):
        i = np.argmax(d, axis=1)
        Ci = np.take_along_axis(Ks, i[:, None, None], axis=1)
        d = np.minimum(d, np.sqrt(((Ks - Ci) ** 2).sum(-1)))
        out.append(d.max(1))
    return np.stack(out, 1)      # (n, rmax)


def run(tag, Lb=64, nq=24, local=8, use_rope=True, seed=0, max_cfg=None, hstride=1):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")
    acc = collections.defaultdict(lambda: {"sig": [], "sperp": [], "rmse": [],
                                           "bias": [], **{k: [] for k in KS}})
    byts = {}; rad = []
    cfgs = sorted(packs.items())
    if max_cfg:
        cfgs = cfgs[:max_cfg]
    for (doc, layer), p in cfgs:
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < max(KS) + 4:
                continue
            cand = np.arange(1, hi)
            Ks = np.stack([K[b * Lb:(b + 1) * Lb] for b in cand])      # (n,L,D)
            rad.append(greedy_radii(Ks, min(Lb, 32)))
            F = build(Ks, Lb, D)
            for h in range(kv * g, (kv + 1) * g, hstride):
                Q = Q_all[qpos, h, :].astype(np.float32)
                Tm = lse(s * np.einsum("nld,jd->nlj", Ks, Q), 1).T
                mass = np.exp(Tm - Tm.max(1, keepdims=True))
                order = np.argsort(-Tm, 1)
                sig_c = Tm - Tm.mean(1, keepdims=True)
                for m, (kind, d, nb) in F.items():
                    Sc = score_family(kind, d, Q, s).T
                    e = Sc - Tm
                    eps = e - e.mean(1, keepdims=True)
                    acc[m]["sig"].append(eps.std(1).mean())
                    acc[m]["rmse"].append(float(np.sqrt((e ** 2).mean())))
                    acc[m]["bias"].append(float(e.mean()))
                    den = (sig_c ** 2).sum(1, keepdims=True)
                    a = (eps * sig_c).sum(1, keepdims=True) / np.maximum(den, 1e-30)
                    acc[m]["sperp"].append(float((eps - a * sig_c).std(1).mean()))
                    om = np.argsort(-Sc, 1)
                    for k in KS:
                        got = np.take_along_axis(mass, om[:, :k], 1).sum(1)
                        ora = np.take_along_axis(mass, order[:, :k], 1).sum(1)
                        acc[m][k].append(float(np.mean(got / np.maximum(ora, 1e-30))))
            byts.update({m: nb for m, (_, _, nb) in F.items()})
    out = {}
    for m, d in acc.items():
        out[m] = dict(bytes=float(byts[m]), rmse=float(np.mean(d["rmse"])),
                      bias=float(np.mean(d["bias"])), sigma_disc=float(np.mean(d["sig"])),
                      sigma_perp=float(np.mean(d["sperp"])),
                      **{f"r@{k}": float(np.mean(d[k])) for k in KS})
    return out, np.concatenate(rad, 0)


def fit_alpha(pts):
    pts = sorted([(b, s) for b, s in pts if s > 0])
    if len(pts) < 2:
        return None
    a, b = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)
    return float(np.exp(b)), float(-a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="qwen8k")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--nq", type=int, default=24)
    ap.add_argument("--maxcfg", type=int, default=None)
    ap.add_argument("--out", default="frontiere.json")
    ap.add_argument("--rope", default="both")
    ap.add_argument("--heads", type=int, default=1)
    a = ap.parse_args()
    allout = {}; t0 = time.time()
    for tag in a.tags.split(","):
        meta, _ = load(tag)
        print(f"[{tag}] meta={meta}")
        ropes = (True, False) if a.rope == "both" else (a.rope == "true",)
        for rope in ropes:
            res, rad = run(tag, Lb=a.block, nq=a.nq, use_rope=rope, max_cfg=a.maxcfg, hstride=a.heads)
            lbl = f"{tag}|{'RoPE' if rope else 'NoPE'}"
            allout[lbl] = res
            print("=" * 100)
            print(f"FRONTIERE OCTETS-PRECISION  {lbl}  (L={a.block}, t={time.time()-t0:.0f}s)")
            print("=" * 100)
            print(f"{'famille':16s} {'octets':>8s} {'RMSE':>7s} {'sig_disc':>9s} "
                  f"{'sig_perp':>9s} {'r@4':>7s} {'r@8':>7s} {'r@16':>7s}")
            print("-" * 100)
            for m, v in sorted(res.items(), key=lambda kv: kv[1]["bytes"]):
                print(f"{m:16s} {v['bytes']:8.0f} {v['rmse']:7.3f} {v['sigma_disc']:9.4f} "
                      f"{v['sigma_perp']:9.4f} {100*v['r@4']:6.2f}% {100*v['r@8']:6.2f}% "
                      f"{100*v['r@16']:6.2f}%")
            print("-" * 100)
            for fam, pref in (("coreset", "coreset r="), ("cobs", "cobs r="),
                              ("quant", "quant "), ("quantG", "quantG "),
                              ("sub", "sub r="), ("top", "top r=")):
                pts = [(v["bytes"], v["sigma_disc"]) for m, v in res.items() if m.startswith(pref)]
                f = fit_alpha(pts)
                if f:
                    print(f"  {fam:8s} sig_disc = {f[0]:8.4f} b^-{f[1]:.3f} | x2 octets "
                          f"-> /{2**f[1]:.3f} | sigma/2 coûte x{2**(1/f[1]):.1f}")
            # rayons gloutons -> dimension de recouvrement
            dmed = np.median(rad, 0)
            rr = np.arange(1, len(dmed) + 1)
            sel = (rr >= 2) & (rr <= 16) & (dmed > 0)
            if sel.sum() > 2:
                a2, b2 = np.polyfit(np.log(rr[sel]), np.log(dmed[sel]), 1)
                print(f"  rayons gloutons d_r ~ r^{a2:.3f}  -> dimension de recouvrement "
                      f"d_cov = {-1/a2:.2f}  (d_1={dmed[0]:.2f}, d_8={dmed[7]:.3f}, "
                      f"d_16={dmed[15]:.3f})")
            print()
    fp = OUT / a.out
    prev = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    prev.update(allout)
    fp.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    allout = prev
    print("->", fp, f"({time.time()-t0:.0f}s)")
