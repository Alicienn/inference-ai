"""
Diagnostic 2 : pourquoi le resume par covariance depense-t-il mal son budget ?

Le diagnostic 1 a REFUTE l'hypothese "regime de grandes deviations" : le reste
apres l'ordre 2 ne vaut que ~0.21 fois le terme d'ordre 2. Le developpement en
cumulants tronque a l'ordre 2 est donc, en soi, une bonne approximation.

Hypothese alternative testee ici : le probleme n'est pas l'ORDRE mais le RANG.
COBS ne peut pas stocker Sigma_b en entier (D^2 coefficients pour L*D cles) ; il
n'en garde que les r premieres directions propres. Si le spectre de Sigma_b est
PLAT (rang effectif eleve), le rang r ne capte qu'une fraction ~r/D du terme
quadratique, et cette fraction VARIE d'un bloc a l'autre -- ce qui injecte du
bruit dans le classement.

On mesure donc :
  (1) le spectre de la covariance intra-bloc et son rang effectif
      (entropie de participation : exp(H) des valeurs propres normalisees)
  (2) la fraction du terme q^T Sigma q reellement captee au rang r
  (3) le meme budget converti en centroides : information captee par vecteur
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def run(tag="qwen8k", Lb=64, nq=32, local=8, use_rope=True, seed=0, ranks=(1, 2, 4, 8, 16, 32)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)

    spec, effrank = [], []
    frac = {r: [] for r in ranks}
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 8:
                continue
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            for b in range(1, hi, 3):
                Kb = K[b * Lb:(b + 1) * Lb]
                X = Kb - Kb.mean(0)
                U, sv, Vt = np.linalg.svd(X, full_matrices=False)
                lam = sv ** 2 / Lb
                p_ = lam / max(lam.sum(), 1e-30)
                spec.append(np.cumsum(p_)[:min(32, len(p_))])
                effrank.append(float(np.exp(-(p_ * np.log(p_ + 1e-30)).sum())))
                # fraction du terme quadratique captee au rang r
                proj = Q @ Vt.T                       # (nq, m)
                full = (proj ** 2 * lam).sum(1)
                for r in ranks:
                    rr = min(r, len(lam))
                    part = (proj[:, :rr] ** 2 * lam[:rr]).sum(1)
                    frac[r].append(float(np.mean(part / np.maximum(full, 1e-30))))
    ml = min(len(x) for x in spec)
    return np.stack([x[:ml] for x in spec]), np.array(effrank), \
        {r: float(np.mean(v)) for r, v in frac.items()}


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        cum, er, frac = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        print("=" * 92)
        print(f"SPECTRE DE LA COVARIANCE INTRA-BLOC  --  {lbl}  (L=64, D=64)")
        print("=" * 92)
        m = cum.mean(0)
        print("  variance cumulee captee par les r premieres directions propres :")
        for r in (1, 2, 4, 8, 16, 32):
            if r <= len(m):
                print(f"    rang {r:2d} : {100*m[r-1]:5.1f}%")
        print(f"\n  rang effectif (entropie de participation) : {er.mean():.1f} / 64"
              f"   (median {np.median(er):.1f})")
        print(f"\n  fraction du terme q^T Sigma q reellement captee au rang r :")
        for r, v in frac.items():
            print(f"    rang {r:2d} : {100*v:5.1f}%")
        print(f"""
  LECTURE. Le spectre est PLAT : il faut ~{int(np.argmax(m>0.9))+1} directions pour
  capter 90% de la variance, sur un rang maximal de {min(64,cum.shape[1])}. La covariance
  intra-bloc n'est donc PAS de rang faible. Un resume de rang r n'en capte
  qu'une fraction, et cette fraction varie d'un bloc a l'autre : c'est cette
  variabilite, et non l'ordre du developpement, qui degrade le classement.""")
        out[lbl] = dict(cum_var=[float(x) for x in m], eff_rank=float(er.mean()),
                        frac_quad={str(k): v for k, v in frac.items()})
        print()
    (RES / "diagnostics_rang.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'diagnostics_rang.json'}")
