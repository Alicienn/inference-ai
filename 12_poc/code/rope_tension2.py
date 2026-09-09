"""
FRONT 3, suite — mon cadrage « contenu » était faux. Diagnostic et balayage.

CE QUE LA PREMIERE MESURE A REFUTE. J'avais defini phi(L), la fraction d'energie des
cles dont la similarite de CONTENU survit a RoPE, et predit qu'elle gouvernerait le gain
du decoupage par contenu. Mesure : phi = 0,734 pour Qwen contre 0,121 pour SmolLM2, alors
que SmolLM2 gagne DAVANTAGE (x1,37 contre x1,23). **phi ne predit pas le gain.**

REFORMULATION. L'objectif n'a jamais ete la coherence semantique : c'est la COMPACITE
GEOMETRIQUE dans l'espace ou l'attention opere reellement, c'est-a-dire l'espace
POST-RoPE. C'est le rayon rho qui borne l'erreur des resumes, et rho se mesure la.
Regrouper avant rotation optimise la compacite dans le mauvais espace — ce qui explique
directement pourquoi le pre-RoPE est moins bon (x1,08-1,10 contre x1,21-1,37).
« Blocs definis par le contenu » est donc un abus de langage : il faut dire
**blocs definis par la geometrie des cles effectivement mises en cache**.

PREDICTION TESTABLE DE CETTE REFORMULATION. Si les groupes post-RoPE optimisent la
compacite geometrique et non la semantique, alors quand RoPE domine la geometrie (phi
faible, cas SmolLM2) les groupes doivent etre structures par la PHASE de rotation, donc
presenter une structure POSITIONNELLE (periodique ou locale) et non aleatoire. On mesure
l'etalement positionnel des groupes et on le compare a un tirage aleatoire.

On balaie par ailleurs la taille et le nombre de blocs, pour savoir si le gain mesure
(x2,0 sans RoPE) est robuste ou specifique au point L=64.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load
from content_blocks import balanced_kmeans as _bk_slow

def balanced_kmeans_fast(K, nb, size, iters=8, rounds=6, seed=0):
    """
    k-means a capacite contrainte, VECTORISE.
    L'affectation gloutonne point par point de content_blocks.balanced_kmeans est en
    O(n*nb*log nb) avec une boucle Python : redhibitoire des que nb depasse ~100.
    Ici on procede par TOURS : (1) chaque point non affecte va au plus proche cluster
    encore ouvert ; (2) tout cluster surcharge ne garde que ses `size` points les plus
    proches, les autres repartent au tour suivant. Converge en quelques tours et reste
    entierement vectorise.
    """
    n, D = K.shape
    rng = np.random.default_rng(seed)
    C = K[rng.choice(n, nb, replace=False)].copy()
    lab = np.full(n, -1, dtype=np.int64)
    for _ in range(iters):
        d = ((K[:, None, :] - C[None]) ** 2).sum(-1)          # (n, nb)
        lab[:] = -1
        cap = np.full(nb, size, dtype=np.int64)
        openc = np.ones(nb, dtype=bool)
        for _r in range(rounds):
            free = np.where(lab < 0)[0]
            if free.size == 0 or not openc.any():
                break
            dd = np.where(openc[None, :], d[free], np.inf)
            cand = dd.argmin(1)
            for j in np.unique(cand):
                pts = free[cand == j]
                if pts.size > cap[j]:
                    keep = pts[np.argsort(d[pts, j])[:cap[j]]]
                else:
                    keep = pts
                lab[keep] = j
                cap[j] -= keep.size
                if cap[j] <= 0:
                    openc[j] = False
        rest = np.where(lab < 0)[0]
        if rest.size:                                          # reliquat : plus proche
            lab[rest] = d[rest].argmin(1)
        for j in range(nb):
            m = lab == j
            if m.any():
                C[j] = K[m].mean(0)
    return lab

balanced_kmeans = balanced_kmeans_fast

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def slow_idx(D, base, L, tau):
    i = np.arange(D // 2)
    th = base ** (-2.0 * i / D)
    sel = np.where(L * th < tau)[0]
    return (np.concatenate([sel, sel + D // 2]) if len(sel) else None), sel


def pos_structure(blocks, ia):
    """Etalement positionnel des groupes, normalise par celui d'un tirage aleatoire."""
    sp, per = [], []
    lo, hi = ia.min(), ia.max()
    for ix in blocks:
        if len(ix) < 4:
            continue
        sp.append(float(np.std(ix)))
        d = np.diff(np.sort(ix))
        per.append(float(np.median(d)))
    rnd = (hi - lo) / np.sqrt(12)              # ecart-type d'un uniforme
    return float(np.mean(sp) / rnd), float(np.mean(per))


def run(tag, base, nq=32, local=8, topk=8, seed=0,
        Lbs=(32, 64, 128, 256), taus=(1.0, 2.0, 4.0, 8.0, 16.0)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    r = 4

    # ---------- (A) balayage de la taille de bloc ----------
    print("=" * 100)
    print(f"(A) BALAYAGE DE LA TAILLE DE BLOC — {tag}, coreset r={r}, top-{topk}")
    print("=" * 100)
    print(f"{'L':>5s} {'nb blocs':>9s} | {'rho pos':>8s} {'rho cont':>9s} | "
          f"{'masse pos':>10s} {'masse cont':>11s} {'GAIN':>7s} | {'sig pos':>8s} {'sig cont':>9s}")
    print("-" * 100)
    sweepA = []
    for Lb in Lbs:
        agg = {m: dict(abs=[], rho=[], sig=[]) for m in ("position", "contenu")}
        for (doc, layer), p in sorted(packs.items())[::2]:
            Ka, Qa = p["k_rope"], p["q_rope"]
            for kv in range(KVH):
                K = Ka[:, kv, :].astype(np.float32)
                qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
                hi = int(qpos.min()) // Lb - local
                if hi < 4 * topk:
                    continue
                ia = np.arange(Lb, hi * Lb); nb = len(ia) // Lb
                ia = ia[:nb * Lb]
                lab = balanced_kmeans(K[ia], nb, Lb, seed=seed)
                parts = {"position": [ia[b * Lb:(b + 1) * Lb] for b in range(nb)],
                         "contenu": [x for x in (ia[lab == j] for j in range(nb))
                                     if len(x) > 4]}
                Q = Qa[qpos, kv * g, :].astype(np.float32)
                for mode, blocks in parts.items():
                    Tm = np.stack([S.true_logmass(K[ix], Q, s) for ix in blocks], 1)
                    mass = np.exp(Tm - Tm.max(1, keepdims=True))
                    share = mass / mass.sum(1, keepdims=True)
                    sums = [S.s_coreset(K[ix], r)[0] for ix in blocks]
                    Sc = np.stack([S.score(su, Q, s) for su in sums], 1)
                    e = Sc - Tm; eps = e - e.mean(1, keepdims=True)
                    om = np.argsort(-Sc, 1)[:, :topk]
                    agg[mode]["abs"].append(float(np.mean(
                        np.take_along_axis(share, om, 1).sum(1))))
                    agg[mode]["sig"].append(float(eps.std(1).mean()))
                    agg[mode]["rho"].append(float(np.mean(
                        [S.max_radius(K[ix], S.kcenter(K[ix], r)) for ix in blocks[:12]])))
        if not agg["position"]["abs"]:
            continue
        ap, ac = np.mean(agg["position"]["abs"]), np.mean(agg["contenu"]["abs"])
        rp, rc = np.mean(agg["position"]["rho"]), np.mean(agg["contenu"]["rho"])
        sp_, sc_ = np.mean(agg["position"]["sig"]), np.mean(agg["contenu"]["sig"])
        print(f"{Lb:5d} {int((T*0.75)//Lb):9d} | {rp:8.2f} {rc:9.2f} | {100*ap:9.2f}% "
              f"{100*ac:10.2f}% {ac/ap:6.2f}x | {sp_:8.3f} {sc_:9.3f}")
        sweepA.append(dict(Lb=Lb, abs_pos=ap, abs_cont=ac, gain=ac / ap,
                           rho_pos=rp, rho_cont=rc, sig_pos=sp_, sig_cont=sc_))

    # ---------- (B) balayage du budget top-k ----------
    print(f"\n(B) BALAYAGE DU BUDGET top-k  (L=64)")
    print(f"{'k':>4s} {'masse pos':>10s} {'masse cont':>11s} {'GAIN':>7s}")
    print("-" * 100)
    sweepB = []
    Lb = 64
    agg = {m: {k: [] for k in (2, 4, 8, 16, 32)} for m in ("position", "contenu")}
    for (doc, layer), p in sorted(packs.items())[::2]:
        Ka, Qa = p["k_rope"], p["q_rope"]
        for kv in range(KVH):
            K = Ka[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 40:
                continue
            ia = np.arange(Lb, hi * Lb); nb = len(ia) // Lb
            ia = ia[:nb * Lb]
            lab = balanced_kmeans(K[ia], nb, Lb, seed=seed)
            parts = {"position": [ia[b * Lb:(b + 1) * Lb] for b in range(nb)],
                     "contenu": [x for x in (ia[lab == j] for j in range(nb)) if len(x) > 4]}
            Q = Qa[qpos, kv * g, :].astype(np.float32)
            for mode, blocks in parts.items():
                Tm = np.stack([S.true_logmass(K[ix], Q, s) for ix in blocks], 1)
                mass = np.exp(Tm - Tm.max(1, keepdims=True))
                share = mass / mass.sum(1, keepdims=True)
                sums = [S.s_coreset(K[ix], r)[0] for ix in blocks]
                Sc = np.stack([S.score(su, Q, s) for su in sums], 1)
                om = np.argsort(-Sc, 1)
                for k in (2, 4, 8, 16, 32):
                    agg[mode][k].append(float(np.mean(
                        np.take_along_axis(share, om[:, :k], 1).sum(1))))
    for k in (2, 4, 8, 16, 32):
        ap, ac = np.mean(agg["position"][k]), np.mean(agg["contenu"][k])
        print(f"{k:4d} {100*ap:9.2f}% {100*ac:10.2f}% {ac/ap:6.2f}x")
        sweepB.append(dict(k=k, abs_pos=ap, abs_cont=ac, gain=ac / ap))

    # ---------- (C) structure positionnelle + sous-espace lent ----------
    print(f"\n(C) STRUCTURE DES GROUPES et SOUS-ESPACE LENT")
    print(f"{'schema':>22s} {'etalement/aleatoire':>20s} {'masse captee':>13s} {'gain':>7s}")
    print("-" * 100)
    Lb = 64
    resC = []
    p0 = sorted(packs.items())[0][1]
    K = p0["k_rope"][:, 0, :].astype(np.float32)
    Qa = p0["q_rope"]
    qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
    hi = int(qpos.min()) // Lb - local
    ia = np.arange(Lb, hi * Lb); nb = len(ia) // Lb; ia = ia[:nb * Lb]
    Q = Qa[qpos, 0, :].astype(np.float32)
    schemes = {"position": None, "contenu-postRoPE": np.arange(D)}
    for tau in taus:
        idx, sel = slow_idx(D, base, T, tau)
        if idx is not None and len(sel) >= 2:
            schemes[f"lent tau={tau:g} ({len(sel)}/{D//2})"] = idx
    base_abs = None
    for nm, idx in schemes.items():
        if idx is None:
            blocks = [ia[b * Lb:(b + 1) * Lb] for b in range(nb)]
        else:
            lab = balanced_kmeans(K[ia][:, idx], nb, Lb, seed=seed)
            blocks = [x for x in (ia[lab == j] for j in range(nb)) if len(x) > 4]
        Tm = np.stack([S.true_logmass(K[ix], Q, s) for ix in blocks], 1)
        mass = np.exp(Tm - Tm.max(1, keepdims=True))
        share = mass / mass.sum(1, keepdims=True)
        sums = [S.s_coreset(K[ix], r)[0] for ix in blocks]
        Sc = np.stack([S.score(su, Q, s) for su in sums], 1)
        a = float(np.mean(np.take_along_axis(share, np.argsort(-Sc, 1)[:, :topk], 1).sum(1)))
        spread, _ = pos_structure(blocks, ia)
        if base_abs is None:
            base_abs = a
        print(f"{nm:>22s} {spread:19.3f} {100*a:12.2f}% {a/base_abs:6.2f}x")
        resC.append(dict(scheme=nm, spread=spread, abs=a, gain=a / base_abs))
    print("""
  'etalement/aleatoire' = ecart-type positionnel des groupes rapporte a celui d'un
  tirage uniforme. Valeur ~1 : groupes positionnellement ALEATOIRES. Valeur << 1 :
  groupes positionnellement LOCAUX (donc quasi contigus).""")
    return dict(sweepA=sweepA, sweepB=sweepB, sweepC=resC)


if __name__ == "__main__":
    from transformers import AutoConfig
    out = {}
    for tag, mid in (("qwen8k", "Qwen/Qwen2.5-0.5B"), ("smol8k", "HuggingFaceTB/SmolLM2-135M")):
        cfg = AutoConfig.from_pretrained(mid)
        base = float(getattr(cfg, "rope_theta", None) or 10000.0)
        print(f"\n\n########## {tag}  (base RoPE {base:g}) ##########")
        out[tag] = run(tag, base)
    (RES / "rope_tension2.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RES/'rope_tension2.json'}")
