"""
Coût en rappel de la SÉLECTION STRATIFIÉE — le compromis imposé par la fusion du noyau.

La fusion à un seul lancement (conception C, cf. journal session 8) remplace le top-m
GLOBAL de la passe 1 par une union de tops LOCAUX : les n blocs sont découpés en G
strates, chaque work-group retient le top-(m/G) de sa strate. Zéro synchronisation, mais
la sélection n'est plus exacte.

Hypothèse : les strates étant échangeables (l'ordre des blocs n'a pas de rapport avec leur
pertinence), l'approximation devrait être faible. Risque : si la masse se concentre dans
une strate, le quota m/G la tronque.

On mesure le rappel final APRÈS la passe fine, en faisant varier G, m et le mode de
stratification :
  - `entrelacee` : strate = index modulo G (ce que fait un noyau GPU avec un pas de grille)
  - `contigue`   : strate = tranche contiguë de n/G blocs
Le mode entrelacé devrait être le plus robuste, car il casse toute corrélation entre la
position d'un bloc et sa strate.
"""
import numpy as np, json, pathlib
import summaries as S
from eval_selection import load

RES = pathlib.Path(__file__).resolve().parents[1] / "resultats"
TOPK = 8


def run(tag="qwen8k", Lb=64, nq=48, local=8, use_rope=True, seed=0,
        r1=2, r2=8, ms=(16, 24, 32, 48), Gs=(1, 2, 4, 8, 16, 32)):
    meta, packs = load(tag)
    D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
    s = 1.0 / np.sqrt(D); g = H // KVH
    rng = np.random.default_rng(seed)
    kk, qq = ("k_rope", "q_rope") if use_rope else ("k_pre", "q_pre")
    acc = {mode: {(m, G): [] for m in ms for G in Gs} for mode in ("entrelacee", "contigue")}
    ref = {m: [] for m in ms}

    for (doc, layer), p in sorted(packs.items()):
        K_all, Q_all = p[kk], p[qq]
        for kv in range(KVH):
            K = K_all[:, kv, :].astype(np.float32)
            qpos = rng.choice(np.arange(int(T * 0.75), T), size=nq, replace=False)
            hi = int(qpos.min()) // Lb - local
            if hi < 56:
                continue
            cand = np.arange(1, hi); n = len(cand)
            s1 = [S.s_coreset(K[b * Lb:(b + 1) * Lb], r1)[0] for b in cand]
            s2 = [S.s_coreset(K[b * Lb:(b + 1) * Lb], r2)[0] for b in cand]
            Q = Q_all[qpos, kv * g, :].astype(np.float32)
            Tm = np.stack([S.true_logmass(K[b * Lb:(b + 1) * Lb], Q, s) for b in cand], 1)
            mass = np.exp(Tm - Tm.max(1, keepdims=True))
            order = np.argsort(-Tm, 1)
            ora = np.take_along_axis(mass, order[:, :TOPK], 1).sum(1)
            Sc1 = np.stack([S.score(su, Q, s) for su in s1], 1)
            Sc2 = np.stack([S.score(su, Q, s) for su in s2], 1)

            def final(surv):
                """surv : (nq, m) indices survivants -> rappel de masse apres passe fine"""
                fine = np.take_along_axis(Sc2, surv, 1)
                sel = np.take_along_axis(surv, np.argsort(-fine, 1)[:, :TOPK], 1)
                got = np.take_along_axis(mass, sel, 1).sum(1)
                return float(np.mean(got / np.maximum(ora, 1e-30)))

            for m in ms:
                if m > n:
                    continue
                ref[m].append(final(np.argsort(-Sc1, 1)[:, :m]))     # top-m GLOBAL
                for G in Gs:
                    q_ = max(m // G, 1)
                    for mode in ("entrelacee", "contigue"):
                        parts = []
                        for gi in range(G):
                            idx = (np.arange(gi, n, G) if mode == "entrelacee"
                                   else np.arange(gi * n // G, (gi + 1) * n // G))
                            if len(idx) == 0:
                                continue
                            sub = Sc1[:, idx]
                            k_ = min(q_, len(idx))
                            top = np.argsort(-sub, 1)[:, :k_]
                            parts.append(idx[top])
                        surv = np.concatenate(parts, 1)
                        acc[mode][(m, G)].append(final(surv))
    return acc, ref, ms, Gs


if __name__ == "__main__":
    out = {}
    for rope in (True, False):
        acc, ref, ms, Gs = run(use_rope=rope)
        lbl = "avec RoPE" if rope else "sans RoPE"
        print("=" * 100)
        print(f"COÛT DE LA SÉLECTION STRATIFIÉE  ({lbl}, r1=2 -> r2=8, top-{TOPK})")
        print("=" * 100)
        print(f"  rappel final après la passe fine ; G = nombre de strates = "
              f"work-groups par requête")
        print(f"\n{'m':>4s} {'global':>8s} | " +
              " ".join(f"{'G='+str(G):>8s}" for G in Gs) + "   (mode entrelacé)")
        print("-" * 100)
        res = {}
        for m in ms:
            if not ref[m]:
                continue
            g0 = np.mean(ref[m])
            cells = [np.mean(acc["entrelacee"][(m, G)]) if acc["entrelacee"][(m, G)] else np.nan
                     for G in Gs]
            print(f"{m:4d} {100*g0:7.2f}% | " + " ".join(f"{100*c:7.2f}%" for c in cells))
            res[m] = dict(global_=float(g0),
                          entrelacee={str(G): float(c) for G, c in zip(Gs, cells)})
        print(f"\n{'m':>4s} {'global':>8s} | " +
              " ".join(f"{'G='+str(G):>8s}" for G in Gs) + "   (mode contigu)")
        print("-" * 100)
        for m in ms:
            if not ref[m]:
                continue
            g0 = np.mean(ref[m])
            cells = [np.mean(acc["contigue"][(m, G)]) if acc["contigue"][(m, G)] else np.nan
                     for G in Gs]
            print(f"{m:4d} {100*g0:7.2f}% | " + " ".join(f"{100*c:7.2f}%" for c in cells))
            res[m]["contigue"] = {str(G): float(c) for G, c in zip(Gs, cells)}
        # perte max
        for mode in ("entrelacee", "contigue"):
            worst = min((np.mean(acc[mode][(m, G)]) / np.mean(ref[m])
                         for m in ms for G in Gs
                         if acc[mode][(m, G)] and ref[m]), default=np.nan)
            print(f"\n  perte maximale, mode {mode:12s} : {100*(1-worst):5.2f} points relatifs")
        out[lbl] = res
        print()
    (RES / "stratified.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES/'stratified.json'}")
