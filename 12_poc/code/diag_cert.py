"""
Diagnostic : un certificat de selection est-il seulement POSSIBLE ?

On mesure les trois grandeurs qui decident :
  (a) M = s ||q|| rho : l'amplitude des ecarts intra-cluster, en nats.
      Toute inegalite de concentration (Hoeffding, Bernstein/Bennett) ne devient
      informative que si M est de l'ordre de 1. Si M >> 1, le facteur de Bennett
      g(M) = (e^M - 1 - M)/M^2 explose et la borne est vide.
  (b) Delta = l_b - lhat_b : l'ecart REEL entre la vraie log-masse et celle du
      coreset. Sa MOYENNE est un biais (sans effet sur le classement) ; c'est son
      ECART-TYPE ENTRE BLOCS qui limite la discrimination.
  (c) l'ecart de score entre le k-ieme et le (k+1)-ieme bloc : le "signal" a
      depasser pour pouvoir eliminer un bloc par preuve.

Si sigma(Delta) << ecart typique, un certificat CALIBRE (probabiliste) est
possible meme si le certificat pire-cas ne l'est pas.
On teste aussi si Delta est PREVISIBLE a partir de statistiques stockables :
si oui, on peut corriger le biais par bloc et ameliorer le CLASSEMENT.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load
from certified import summary_cert, bennett_g

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0, r=4, topk=8):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    Ms, Deltas, gaps = [], [], []
    feats, targs = [], []

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 16:
                continue
            cand = np.arange(1, hi)
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            qn = np.linalg.norm(Q, axis=1)

            lh, tm, Mb, feat = [], [], [], []
            for b in cand:
                Kb = K[b * Lb:(b + 1) * Lb]
                su, _ = summary_cert(Kb, r)
                A = s * (Q @ su["mu"].T) + np.log(su["n"])[None, :]
                lh.append(S._lse(A))
                tm.append(S.true_logmass(Kb, Q, s))
                Mb.append(s * qn * su["rho"].max())
                # descripteurs stockables : rayon max, vp max, log L
                feat.append(np.stack([
                    s * qn * su["rho"].max(),
                    (s * qn) ** 2 * su["lam"].max(),
                    (s * qn) ** 2 * float(su["lam"].mean()),
                    np.full(len(Q), np.log(Lb)),
                ], 1))
            lh = np.stack(lh, 1); tm = np.stack(tm, 1); Mb = np.stack(Mb, 1)
            d = tm - lh
            Ms.append(Mb.ravel()); Deltas.append(d.ravel())
            srt = np.sort(tm, 1)[:, ::-1]
            gaps.append((srt[:, topk - 1] - srt[:, topk]))
            F = np.stack(feat, 1)                     # (nq, nb, 4)
            feats.append(F.reshape(-1, F.shape[-1])); targs.append(d.ravel())

    return (np.concatenate(Ms), np.concatenate(Deltas), np.concatenate(gaps),
            np.concatenate(feats), np.concatenate(targs))


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        M, Dl, gap, F, y = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        print("=" * 96)
        print(f"UN CERTIFICAT EST-IL POSSIBLE ?  --  {lbl}  (coreset r=4, L=64)")
        print("=" * 96)
        print(f"(a) amplitude des ecarts  M = s||q||rho :")
        print(f"      median {np.median(M):8.2f} nats   p90 {np.percentile(M,90):8.2f}")
        print(f"      facteur de Bennett g(M) median : {bennett_g(np.median(M)):.3e}")
        print(f"      -> une borne de concentration n'est informative que si M ~ 1.")
        print(f"\n(b) ecart reel Delta = l_b - lhat_b :")
        print(f"      moyenne (biais) {Dl.mean():7.3f}   ecart-type ENTRE BLOCS "
              f"{Dl.std():7.3f} nats")
        print(f"      max {Dl.max():7.3f}   min {Dl.min():7.3f}")
        print(f"\n(c) ecart de score entre le {8}e et le {9}e bloc :")
        print(f"      median {np.median(gap):7.3f}   p25 {np.percentile(gap,25):7.3f}"
              f"   p75 {np.percentile(gap,75):7.3f} nats")
        ratio = Dl.std() / max(np.median(gap), 1e-9)
        print(f"\n  RAPPORT sigma(Delta) / ecart median = {ratio:.2f}")
        if ratio > 1:
            print("  -> l'incertitude residuelle depasse l'ecart a discriminer :")
            print("     aucun certificat, meme calibre, ne peut trancher de facon fiable.")
        else:
            print("  -> l'incertitude est INFERIEURE a l'ecart : un certificat calibre")
            print("     (probabiliste) est possible, meme si le pire cas est vide.")

        # Delta est-il previsible a partir de statistiques stockables ?
        A = np.concatenate([F, np.ones((len(F), 1))], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        print(f"\n(d) Delta est-il PREVISIBLE a partir de (rho, lambda) stockes ?")
        print(f"      R^2 de la regression = {r2:.3f}   "
              f"residu sigma = {np.std(y - pred):.3f} nats (contre {Dl.std():.3f} brut)")
        if r2 > 0.3:
            print("      -> oui : on peut corriger le biais PAR BLOC et esperer")
            print("         ameliorer le classement, pas seulement l'estimation.")
        else:
            print("      -> non : le biais n'est pas explique par ces statistiques.")
        out[lbl] = dict(M_median=float(np.median(M)), delta_mean=float(Dl.mean()),
                        delta_std=float(Dl.std()), gap_median=float(np.median(gap)),
                        ratio=float(ratio), r2=float(r2),
                        resid=float(np.std(y - pred)), coef=[float(c) for c in coef])
        print()
    (RES / "diag_cert.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'diag_cert.json'}")
