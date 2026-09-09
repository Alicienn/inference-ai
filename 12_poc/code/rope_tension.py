"""
FRONT 3 — la tension RoPE / découpage par contenu, formalisée puis exploitée.

=======================================================================================
1. LA TENSION, DÉRIVÉE
=======================================================================================
[DÉRIVATION] Convention Llama : la dimension de tête d se découpe en d/2 blocs de
fréquence 2D, le bloc i associant les dimensions (i, i+d/2) et tournant à la vitesse
theta_i = base^(-2i/d).

Prenons deux tokens de contenu IDENTIQUE (même clé pré-RoPE k) aux positions p et p'.
Après rotation, leur distance vaut

    ||R(p)k - R(p')k||^2 = 2 * somme_i ||k_i||^2 * (1 - cos(Delta * theta_i)),  Delta = p'-p

Autrement dit : **RoPE crée de la dispersion là où il n'y a aucune différence de
contenu.** C'est exactement ce que le découpage par contenu cherche à éviter, puisque
le rayon intra-bloc rho borne l'erreur de tout résumé (théorème du POC).

En moyennant sur Delta uniforme sur [0, L] (L = longueur de contexte) :

    E[1 - cos(Delta*theta_i)] = 1 - sinc(L*theta_i),   sinc(x) = sin(x)/x

Deux régimes nets :
  - L*theta_i << 1 : sinc ~ 1 - (L theta_i)^2/6, dispersion ~ ||k_i||^2 (L theta_i)^2/3.
    Négligeable : la similarité de contenu SURVIT.
  - L*theta_i >> 1 : sinc ~ 0, dispersion ~ 2||k_i||^2. Le contenu est DÉTRUIT.

D'où une grandeur unique qui résume la compatibilité d'un modèle avec le découpage
par contenu — nous l'appelons la **fraction de contenu préservée** :

    phi(L) = somme_i  E_i * sinc(L*theta_i)  /  somme_i E_i        avec E_i = ||k_i||^2

phi = 1 : RoPE n'entrave rien (ou pas de RoPE). phi = 0 : contenu entièrement brouillé.

[INFÉRENCE] phi dépend de l'ÉNERGIE PAR FRÉQUENCE E_i, mesurée en session 2 : elle est
concentrée sur les basses fréquences (ratio 35x). Or ce sont précisément celles qui
tournent lentement. **L'énergie est là où la rotation est lente** — c'est pourquoi le
découpage par contenu fonctionne encore, partiellement, malgré RoPE.

=======================================================================================
2. LE SCHÉMA QUI EN DÉCOULE  [HYPOTHÈSE/PROPOSITION]
=======================================================================================
Si seules les basses fréquences préservent le contenu, il faut **regrouper sur le
sous-espace lent uniquement**, en ignorant les dimensions rapides qui n'apportent que
du bruit rotationnel. On définit le sous-espace robuste

    I(tau) = { i : L * theta_i < tau }          (tau ~ 1, seuil de décorrélation)

et l'on effectue le k-means sur la projection des clés post-RoPE sur ces dimensions.
Ce n'est ni RoPE classique (qui utilise tout) ni NoPE (qui supprime tout) : c'est un
**découpage sélectif en fréquence**, qui garde la qualité d'attention intacte (on ne
touche pas au modèle) et ne change que le critère de PARTITION.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load
from content_blocks import balanced_kmeans

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def phi(L, D, base, E=None):
    """Fraction de contenu préservée."""
    i = np.arange(D // 2)
    th = base ** (-2.0 * i / D)
    E = np.ones(D // 2) if E is None else E
    x = np.maximum(L * th, 1e-12)
    return float((E * np.sinc(x / np.pi)).sum() / E.sum()), th


def slow_dims(D, base, L, tau=1.0):
    """Indices de dimensions appartenant au sous-espace lent I(tau)."""
    i = np.arange(D // 2)
    th = base ** (-2.0 * i / D)
    sel = np.where(L * th < tau)[0]
    return np.concatenate([sel, sel + D // 2]), sel


def measure_energy(packs, kv, kk, D):
    """Profil d'énergie par bloc de fréquence, mesuré sur les clés réelles."""
    E = np.zeros(D // 2); n = 0
    for (_, _), p in sorted(packs.items()):
        K = p[kk][:, kv, :].astype(np.float32)
        h = D // 2
        E += (K[:, :h] ** 2 + K[:, h:] ** 2).mean(0); n += 1
    return E / max(n, 1)


def run(tag, base, Lb=64, nq=32, local=8, topk=8, seed=0, rs=(1, 4)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    E = measure_energy(packs, 0, "k_pre", D)
    ph_u, th = phi(T, D, base)
    ph_e, _ = phi(T, D, base, E)
    idx_slow, sel = slow_dims(D, base, T, tau=1.0)
    print(f"\n{'='*100}\n{tag} — base RoPE {base:g}, D={D}, contexte {T}\n{'='*100}")
    print(f"  phi (énergie uniforme)     = {ph_u:.3f}")
    print(f"  phi (énergie MESURÉE)      = {ph_e:.3f}   <- l'énergie est sur les basses fréquences")
    print(f"  sous-espace lent I(tau=1)  : {len(sel)}/{D//2} blocs de fréquence "
          f"({100*len(sel)/(D//2):.0f} %), soit {len(idx_slow)}/{D} dimensions")
    if len(sel):
        print(f"  part d'énergie dans le sous-espace lent : {100*E[sel].sum()/E.sum():.1f} %")

    modes = ["position", "contenu-postRoPE", "contenu-preRoPE", "contenu-LENT"]
    out = {m: {r: dict(rho=[], sig=[], abs=[]) for r in rs} for m in modes}
    for (doc, layer), p in sorted(packs.items()):
        Ka, Qa, Kp = p["k_rope"], p["q_rope"], p["k_pre"]
        for kv in range(KVH):
            K = Ka[:, kv, :].astype(np.float32)
            Kpre = Kp[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 40:
                continue
            ia = np.arange(Lb, hi * Lb); nb = len(ia) // Lb
            ia = ia[:nb * Lb]
            parts = {"position": [ia[b * Lb:(b + 1) * Lb] for b in range(nb)]}
            for nm, feat in (("contenu-postRoPE", K[ia]),
                             ("contenu-preRoPE", Kpre[ia]),
                             ("contenu-LENT", K[ia][:, idx_slow] if len(idx_slow) else K[ia])):
                lab = balanced_kmeans(feat, nb, Lb, seed=seed)
                parts[nm] = [x for x in (ia[lab == j] for j in range(nb)) if len(x) > 4]
            Q = Qa[qpos, kv * g, :].astype(np.float32)
            for mode, blocks in parts.items():
                Tm = np.stack([S.true_logmass(K[ix], Q, s) for ix in blocks], 1)
                mass = np.exp(Tm - Tm.max(1, keepdims=True))
                share = mass / mass.sum(1, keepdims=True)
                for r in rs:
                    sums = [S.s_coreset(K[ix], r)[0] for ix in blocks]
                    Sc = np.stack([S.score(su, Q, s) for su in sums], 1)
                    e = Sc - Tm; eps = e - e.mean(1, keepdims=True)
                    om = np.argsort(-Sc, 1)[:, :topk]
                    d = out[mode][r]
                    d["sig"].append(float(eps.std(1).mean()))
                    d["abs"].append(float(np.mean(np.take_along_axis(share, om, 1).sum(1))))
                    d["rho"].append(float(np.mean(
                        [S.max_radius(K[ix], S.kcenter(K[ix], r)) for ix in blocks[:16]])))
    print(f"\n{'r':>3s} {'partition':>18s} {'rho':>8s} {'sigma':>8s} "
          f"{'masse ABSOLUE captée':>22s} {'vs position':>12s}")
    print("-" * 100)
    res = {}
    for r in rs:
        base_abs = np.mean(out["position"][r]["abs"])
        for mode in modes:
            d = out[mode][r]
            if not d["abs"]:
                continue
            a = np.mean(d["abs"])
            print(f"{r:3d} {mode:>18s} {np.mean(d['rho']):8.3f} {np.mean(d['sig']):8.3f} "
                  f"{100*a:21.2f}% {a/base_abs:11.2f}x")
            res[f"{mode}_r{r}"] = dict(rho=float(np.mean(d["rho"])),
                                       sig=float(np.mean(d["sig"])), abs=float(a),
                                       gain=float(a / base_abs))
    return dict(phi_unif=ph_u, phi_meas=ph_e, n_slow=int(len(sel)),
                energy_slow=float(E[sel].sum() / E.sum()) if len(sel) else 0.0, res=res)


if __name__ == "__main__":
    from transformers import AutoConfig
    out = {}
    for tag, mid in (("qwen8k", "Qwen/Qwen2.5-0.5B"),
                     ("smol8k", "HuggingFaceTB/SmolLM2-135M")):
        cfg = AutoConfig.from_pretrained(mid)
        base = float(getattr(cfg, "rope_theta", None) or 10000.0)
        out[tag] = run(tag, base)
        out[tag]["rope_theta"] = base
    (RES / "rope_tension.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RES/'rope_tension.json'}")
