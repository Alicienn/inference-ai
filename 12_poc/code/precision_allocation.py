"""
CONCEPT — ASP : Allocation Séquentielle de Précision
=====================================================

HYPOTHÈSE IMPLICITE QUE TOUTE LA LITTÉRATURE PARTAGE
----------------------------------------------------
NSA, Quest, COBS, CSA, AsyncTLS, et mes propres coresets allouent tous le MÊME
budget d'octets à CHAQUE bloc, et lisent chaque résumé exactement une fois. C'est
un schéma à passe unique et à résolution uniforme. Personne ne le questionne.

Or le problème sous-jacent — « trouver les k plus grands parmi n à partir
d'estimations bruitées, sous budget de mesure » — est résolu depuis longtemps
ailleurs : c'est l'IDENTIFICATION DES MEILLEURS BRAS en bandits stochastiques
(Bubeck-Munos-Stoltz 0802.2655 ; Karnin, Sequential Halving ; Carpentier-Locatelli
1605.09004). Sa théorie dit que l'allocation optimale est SÉQUENTIELLE et NON
UNIFORME : on élimine tôt les bras clairement perdants et on concentre le budget
restant sur les prétendants.

UNE VARIANTE NOUVELLE DU PROBLÈME DE BANDIT
--------------------------------------------
La correspondance n'est pas exacte, et c'est ce qui rend le problème intéressant.
En bandit classique, tirer un bras t fois fait décroître le bruit en 1/sqrt(t) par
moyennage stochastique. Ici, lire b octets de résumé fait décroître le bruit selon
une courbe DÉBIT-DISTORSION DÉTERMINISTE sigma(b) — pas de moyennage, pas d'aléa.
Nous appelons cela un **bandit à allocation de précision** : le budget est en bits
par bras, et la précision obéit à une loi de débit-distorsion mesurable.

L'ALGORITHME QUI EN DÉCOULE
----------------------------
  Passe 1 : lire un résumé GROSSIER (b1 octets) de TOUS les n blocs -> lhat^(1)
  Élimination : ne garder que les m blocs dont le score est assez proche du seuil
                pour que le raffinement puisse encore changer la décision.
  Passe 2 : lire un résumé FIN (b2 octets) des m survivants seulement.
  Coût total : n*b1 + m*b2   au lieu de   n*b2.

Le gain vient de ce que m << n : la plupart des blocs sont éliminables avec une
précision grossière, car ils sont loin du seuil.

CE QUE ÇA N'EST PAS. AsyncTLS (2604.07815) fait aussi deux niveaux, mais raffine la
GRANULARITÉ (blocs -> tokens) à résumé fixe. HiSparse (2608.07009) hiérarchise le
STOCKAGE (hôte/GPU). Ici on raffine la PRÉCISION DU RÉSUMÉ à granularité constante,
et on alloue le budget en fonction de la distance au seuil de décision.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"


def build(K, Lb, nb, rs):
    """Résumés coreset de rang r pour tous les blocs, pour chaque r de rs."""
    out = {r: [] for r in rs}
    cost = {}
    for b in range(nb):
        Kb = K[b * Lb:(b + 1) * Lb]
        for r in rs:
            su, c = S.s_coreset(Kb, r)
            out[r].append(su); cost[r] = c
    return out, cost


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0,
        topk=8, rs=(1, 2, 4, 8, 16), ms=(10, 12, 16, 24, 32, 48)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D)
    g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")

    flat = {r: dict(rec=[], byt=[], sig=[]) for r in rs}
    prog = {(m, r): dict(rec=[], byt=[]) for m in ms for r in rs if r >= 4}
    prog3 = {}

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 56:
                continue
            cand = np.arange(1, hi)
            n = len(cand)
            sums, cost = build(K, Lb, hi, rs)
            h = kv * g
            Q = Q_all[qpos, h, :].astype(np.float32)
            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s)
                           for b in cand], 1)
            mass = np.exp(Tm - Tm.max(1, keepdims=True))
            order = np.argsort(-Tm, 1)
            ora = np.take_along_axis(mass, order[:, :topk], 1).sum(1)

            Sc = {}
            for r in rs:
                Sc[r] = np.stack([S.score(sums[r][b], Q, s) for b in cand], 1)
                om = np.argsort(-Sc[r], 1)[:, :topk]
                got = np.take_along_axis(mass, om, 1).sum(1)
                flat[r]["rec"].append(float(np.mean(got / np.maximum(ora, 1e-30))))
                flat[r]["byt"].append(cost[r] * n)      # coût total lu, en vecteurs
                e = Sc[r] - Tm
                eps = e - e.mean(1, keepdims=True)
                flat[r]["sig"].append(float(eps.std(1).mean()))

            # ---- schéma progressif : passe 1 grossière (r=1) puis passe 2 fine
            base = Sc[1]
            keep = np.argsort(-base, 1)                 # (nq, n)
            for m in ms:
                if m > n:
                    continue
                surv = keep[:, :m]
                for r in rs:
                    if r < 4:
                        continue
                    # score fin uniquement sur les survivants
                    fine = np.take_along_axis(Sc[r], surv, 1)
                    sel = np.take_along_axis(surv, np.argsort(-fine, 1)[:, :topk], 1)
                    got = np.take_along_axis(mass, sel, 1).sum(1)
                    prog[(m, r)]["rec"].append(
                        float(np.mean(got / np.maximum(ora, 1e-30))))
                    prog[(m, r)]["byt"].append(cost[1] * n + cost[r] * m)

            # ---- variante à trois passes : r=1 (tous) -> r=4 (m1) -> r=16 (m2)
            for m1, m2 in ((48, 16), (32, 12), (24, 10)):
                if m1 > n:
                    continue
                s1 = keep[:, :m1]
                f1 = np.take_along_axis(Sc[4], s1, 1)
                s2 = np.take_along_axis(s1, np.argsort(-f1, 1)[:, :m2], 1)
                f2 = np.take_along_axis(Sc[16], s2, 1)
                sel = np.take_along_axis(s2, np.argsort(-f2, 1)[:, :topk], 1)
                got = np.take_along_axis(mass, sel, 1).sum(1)
                key = (m1, m2)
                prog3.setdefault(key, dict(rec=[], byt=[]))
                prog3[key]["rec"].append(float(np.mean(got / np.maximum(ora, 1e-30))))
                prog3[key]["byt"].append(cost[1] * n + cost[4] * m1 + cost[16] * m2)
    return flat, prog, prog3


def pareto(points):
    """Points (octets, rappel) non dominés."""
    pts = sorted(points, key=lambda z: z[0])
    best, out = -1, []
    for b, r, lbl in pts:
        if r > best:
            best = r; out.append((b, r, lbl))
    return out


if __name__ == "__main__":
    res = {}
    for rope in (True, False):
        flat, prog, prog3 = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        print("=" * 100)
        print(f"COURBE DÉBIT-DISTORSION DES RÉSUMÉS  ({lbl}, L=64, top-8)")
        print("=" * 100)
        print(f"{'coreset r':>10s} {'vec/bloc':>9s} {'sigma_disc':>11s} {'rappel@8':>10s}")
        print("-" * 100)
        for r, d in sorted(flat.items()):
            n_eff = np.mean(d["byt"]) / (np.mean(d["byt"]) / 1)  # placeholder
            print(f"{r:10d} {np.mean(d['byt'])/87:9.2f} {np.mean(d['sig']):11.3f} "
                  f"{100*np.mean(d['rec']):9.2f}%")

        pts = [(np.mean(d["byt"]), np.mean(d["rec"]), f"plat r={r}")
               for r, d in flat.items()]
        pts += [(np.mean(d["byt"]), np.mean(d["rec"]), f"ASP m={m},r={r}")
                for (m, r), d in prog.items() if d["rec"]]
        pts += [(np.mean(d["byt"]), np.mean(d["rec"]), f"ASP3 {m1}->{m2}")
                for (m1, m2), d in prog3.items() if d["rec"]]

        print()
        print("=" * 100)
        print(f"FRONT DE PARETO  rappel@8 vs OCTETS TOTAUX LUS  ({lbl})")
        print("=" * 100)
        print(f"{'schéma':18s} {'vecteurs lus':>13s} {'/bloc':>8s} {'rappel@8':>10s}")
        print("-" * 100)
        for b, r, nm in pareto(pts):
            print(f"{nm:18s} {b:13.1f} {b/87:8.2f} {100*r:9.2f}%"
                  + ("   <-- ASP" if nm.startswith("ASP") else ""))

        print()
        print("  Comparaison directe à rappel équivalent :")
        for r_t in (0.86, 0.88, 0.895):
            cands = [(b, rr, nm) for b, rr, nm in pts if rr >= r_t]
            if not cands:
                continue
            bf = min([c for c in cands if c[2].startswith("plat")],
                     key=lambda z: z[0], default=None)
            ba = min([c for c in cands if c[2].startswith("ASP")],
                     key=lambda z: z[0], default=None)
            if bf and ba:
                print(f"    rappel >= {100*r_t:.1f}% : plat {bf[0]:7.1f} vec "
                      f"({bf[2]})  |  ASP {ba[0]:7.1f} vec ({ba[2]})"
                      f"  -> **{bf[0]/ba[0]:.2f}x moins d'octets**")
        res[lbl] = dict(
            flat={str(r): dict(bytes=float(np.mean(d["byt"])),
                               sigma=float(np.mean(d["sig"])),
                               rec=float(np.mean(d["rec"]))) for r, d in flat.items()},
            prog={f"{m}_{r}": dict(bytes=float(np.mean(d["byt"])),
                                   rec=float(np.mean(d["rec"])))
                  for (m, r), d in prog.items() if d["rec"]},
            prog3={f"{a}_{b}": dict(bytes=float(np.mean(d["byt"])),
                                    rec=float(np.mean(d["rec"])))
                   for (a, b), d in prog3.items() if d["rec"]})
        print()
    (RES / "precision_allocation.json").write_text(json.dumps(res, indent=2))
    print(f"-> {RES/'precision_allocation.json'}")
