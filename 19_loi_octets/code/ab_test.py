"""A/B : mon chemin vectorise vs summaries.py (S.score), sur les memes blocs/requetes."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\12_poc\code")
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code")
import summaries as S
from eval_selection import load
import frontiere2 as F2

meta, packs = load("qwen8k")
print("meta:", meta)
keys = sorted(packs.keys())
print("configs:", keys[:4], "...", len(keys))
D, H, KVH, T = meta["D"], meta["H"], meta["KVH"], meta["seq"]
s = 1.0 / np.sqrt(D); g = H // KVH

for (doc, layer) in keys[:2]:
    p = packs[(doc, layer)]
    for rope in (True, False):
        kk, qq = ("k_rope", "q_rope") if rope else ("k_pre", "q_pre")
        K = p[kk][:, 0, :].astype(np.float32)
        qpos = np.arange(6000, 6000 + 8)
        Q = p[qq][qpos, 0, :].astype(np.float32)
        cand = np.arange(1, 80)
        Ks = np.stack([K[b * 64:(b + 1) * 64] for b in cand])
        Tm = np.stack([S.true_logmass(Ks[i], Q, s) for i in range(len(cand))], 1)
        print(f"--- cfg=({doc},{layer}) rope={rope} : |K| moy={np.linalg.norm(Ks,axis=2).mean():.2f} "
              f"Tm std={Tm.std():.2f} Tm moy={Tm.mean():.2f}")
        # reference summaries.py
        ref = {}
        for name, fn in (("mean", lambda Kb: S.s_mean(Kb)),
                         ("coreset r=2", lambda Kb: S.s_coreset(Kb, 2)),
                         ("coreset r=8", lambda Kb: S.s_coreset(Kb, 8)),
                         ("cobs r=2", lambda Kb: S.s_cobs(Kb, 2)),
                         ("gmm 2x1", lambda Kb: S.s_gmm(Kb, 2, 1))):
            Sc = np.stack([S.score(fn(Ks[i])[0], Q, s) for i in range(len(cand))], 1)
            ref[name] = Sc
        # mon chemin vectorise
        F = F2.build(Ks, 64, D)
        mine = {}
        for name, (kind, d, nb) in F.items():
            if name in ref:
                mine[name] = F2.score_family(kind, d, Q, s).T
        for name in ref:
            dmax = np.abs(ref[name] - mine[name]).max()
            print(f"    {name:12s} ecart max = {dmax:.3e}   (sigma_disc ref={np.mean((ref[name]-ref[name].mean(1,keepdims=True)).std(1)):.4f})")
