"""
BLOCS DÉFINIS PAR LE CONTENU plutôt que par la POSITION.

Hypothèse implicite testée : toute la littérature (NSA, Quest, COBS, CSA, AsyncTLS,
HiSparse) partitionne le KV cache en blocs CONTIGUS EN POSITION. La justification est
matérielle, jamais sémantique.

Or la masse d'attention n'est pas groupée spatialement (autocorrélation 0,13-0,21 au
décalage 1, mesurée en session 2). Un bloc contigu agrège donc des clés sans rapport,
ce qui gonfle son rayon intra-bloc rho — la quantité même qui contrôle l'erreur de tout
résumé, par la borne

    | l_b(q) - lhat_b(q) |  <=  s ||q|| max_c rho_c

Si les blocs étaient définis par le CONTENU (clés similaires regroupées, une fois au
prefill), rho s'effondrerait et tous les résumés gagneraient en précision à budget égal.

PROTOCOLE. À nombre de blocs et taille de bloc IDENTIQUES, on compare :
  (a) blocs contigus en position           <- l'état de l'art
  (b) blocs définis par le contenu, ÉQUILIBRÉS (k-means à capacité contrainte)
On mesure rho, sigma_disc et le rappel de masse. La causalité est vérifiable : si rho
chute et que sigma chute proportionnellement, le mécanisme est bien celui prévu.

HONNÊTETÉ SUR LE PROTOCOLE. Les blocs par contenu ont un avantage structurel :
regrouper les clés similaires concentre la masse sur moins de blocs, ce qui rend le
top-k mécaniquement plus facile. On rapporte donc AUSSI la masse absolue captée par les
k meilleurs blocs (et pas seulement le rappel relatif à l'oracle), car c'est la seule
métrique comparable entre deux partitionnements différents.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def balanced_kmeans(K, nb, size, iters=12, seed=0):
    """
    k-means à capacité contrainte : nb clusters de `size` éléments exactement.
    Affectation gloutonne par distance croissante (approximation raisonnable du
    problème d'affectation, suffisante ici).
    """
    n, D = K.shape
    rng = np.random.default_rng(seed)
    C = K[rng.choice(n, nb, replace=False)].copy()
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((K[:, None, :] - C[None]) ** 2).sum(-1)         # (n, nb)
        order = np.argsort(d.min(1))                          # les plus sûrs d'abord
        cap = np.full(nb, size)
        lab[:] = -1
        for i in order:
            for j in np.argsort(d[i]):
                if cap[j] > 0:
                    lab[i] = j; cap[j] -= 1; break
        for j in range(nb):
            m = lab == j
            if m.any():
                C[j] = K[m].mean(0)
    return lab


def radius(K, idx, r):
    """Rayon max de la partition en r clusters du sous-ensemble idx."""
    Kb = K[idx]
    lab = S.kcenter(Kb, r)
    return S.max_radius(Kb, lab)


def run(tag="qwen8k", Lb=64, nq=32, local=8, use_rope=True, seed=0, topk=8,
        rs=(1, 2, 4, 8)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    MODES = ("position", "contenu", "contenu-preRoPE")
    out = {mode: {r: dict(sig=[], rec=[], abs=[], rho=[]) for r in rs}
           for mode in MODES}

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 40:
                continue
            # univers de clés : les blocs candidats (on exclut puits et fenêtre locale)
            lo_t, hi_t = Lb, hi * Lb
            idx_all = np.arange(lo_t, hi_t)
            Ku = K[idx_all]
            nb = len(idx_all) // Lb
            Ku = Ku[:nb * Lb]; idx_all = idx_all[:nb * Lb]

            parts = {}
            parts["position"] = [idx_all[b * Lb:(b + 1) * Lb] for b in range(nb)]
            lab = balanced_kmeans(Ku, nb, Lb, seed=seed)
            parts["contenu"] = [x for x in (idx_all[lab == j] for j in range(nb))
                                if len(x) > 4]
            # variante pratique : regrouper sur le contenu PRE-RoPE (independant de
            # la position), mais attendre sur les cles POST-RoPE reelles.
            if use_rope:
                Kpre = p["k_pre"][:, kv, :].astype(np.float32)[idx_all]
                lab2 = balanced_kmeans(Kpre, nb, Lb, seed=seed)
                parts["contenu-preRoPE"] = [x for x in
                                            (idx_all[lab2 == j] for j in range(nb))
                                            if len(x) > 4]

            Q = Q_all[qpos, kv * g, :].astype(np.float32)
            for mode, blocks in parts.items():
                Tm = np.stack([S.true_logmass(K[ix], Q, s) for ix in blocks], 1)
                mass = np.exp(Tm - Tm.max(1, keepdims=True))
                share = mass / mass.sum(1, keepdims=True)
                order = np.argsort(-Tm, 1)
                ora = np.take_along_axis(mass, order[:, :topk], 1).sum(1)
                abs_ora = np.take_along_axis(share, order[:, :topk], 1).sum(1)
                for r in rs:
                    sums = [S.s_coreset(K[ix], r)[0] for ix in blocks]
                    Sc = np.stack([S.score(su, Q, s) for su in sums], 1)
                    e = Sc - Tm
                    eps = e - e.mean(1, keepdims=True)
                    om = np.argsort(-Sc, 1)[:, :topk]
                    got = np.take_along_axis(mass, om, 1).sum(1)
                    d = out[mode][r]
                    d["sig"].append(float(eps.std(1).mean()))
                    d["rec"].append(float(np.mean(got / np.maximum(ora, 1e-30))))
                    d["abs"].append(float(np.mean(
                        np.take_along_axis(share, om, 1).sum(1))))
                    d["rho"].append(float(np.mean([radius(K, ix, r) for ix in blocks[:20]])))
    return out


if __name__ == "__main__":
    res = {}
    for tag in ("qwen8k", "smol8k"):
        for rope in (True, False):
            o = run(tag, use_rope=rope)
            lbl = f"{tag} {'avec' if rope else 'sans'} RoPE"
            print("=" * 104)
            print(f"BLOCS PAR CONTENU vs PAR POSITION   ({lbl}, L=64, top-8)")
            print("=" * 104)
            print(f"{'r':>3s} {'partition':>10s} {'rho moyen':>10s} {'sigma_disc':>11s}"
                  f" {'rappel rel.':>12s} {'masse ABSOLUE captée':>22s}")
            print("-" * 104)
            for r in sorted(o["position"]):
                for mode in [m for m in o if o[m][r]["sig"]]:
                    d = o[mode][r]
                    print(f"{r:3d} {mode:>10s} {np.mean(d['rho']):10.3f} "
                          f"{np.mean(d['sig']):11.3f} {100*np.mean(d['rec']):11.2f}% "
                          f"{100*np.mean(d['abs']):21.2f}%")
                rp, rc = np.mean(o["position"][r]["rho"]), np.mean(o["contenu"][r]["rho"])
                sp, sc = np.mean(o["position"][r]["sig"]), np.mean(o["contenu"][r]["sig"])
                ap, ac = np.mean(o["position"][r]["abs"]), np.mean(o["contenu"][r]["abs"])
                print(f"    -> rho x{rc/rp:.2f}   sigma x{sc/sp:.2f}   "
                      f"masse absolue x{ac/ap:.2f}")
            res[lbl] = {m: {str(r): {k: float(np.mean(v)) for k, v in d.items()}
                            for r, d in o[m].items() if d["sig"]} for m in o}
            print()
    (RES / "content_blocks.json").write_text(json.dumps(res, indent=2))
    print(f"-> {RES/'content_blocks.json'}")
