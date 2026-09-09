import sys, numpy as np
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\12_poc\code")
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code")
import summaries as S
from eval_selection import load
import frontiere2 as F2

meta, packs = load("qwen8k")
D = meta["D"]; s = 1.0/np.sqrt(D)
p = packs[(0,2)]
K = p["k_pre"][:, 0, :].astype(np.float32)
qpos = np.arange(6000, 6004)
Q = p["q_pre"][qpos, 0, :].astype(np.float32)
Kb = K[64:128]
Ks = Kb[None]                     # (1,64,64)

lab_ref = S.kcenter(Kb, 2)
lab_mine = F2.kcenter_lab(Ks, 2)[0]
print("labels ref :", lab_ref)
print("labels mine:", lab_mine)
print("meme partition (a permutation pres) ?", sorted(lab_ref.tolist()) == sorted(lab_mine.tolist()))
su_ref, cost_ref = S.s_coreset(Kb, 2)
print("ref  mu=", np.round(su_ref["mu"],3).tolist(), "n=", su_ref["n"].tolist(), "rho=", round(su_ref["rho"],3))
F = F2.build(Ks, 64, D)
kind, dd, nb = F["coreset r=2"]
print("mine mu=", np.round(dd["mu"][0],3).tolist(), "cnt=", dd["cnt"][0].tolist())
print("ref  score:", np.round(S.score(su_ref, Q, s),4).tolist())
print("mine score:", np.round(F2.score_family(kind, dd, Q, s).T[0],4).tolist())
print("exact      :", np.round(S.true_logmass(Kb, Q, s),4).tolist())
# kcenter: la fonction de S.kcenter reutilise-t-elle les centres ou les labels ?
C = np.stack([Kb[i] for i in [int(np.argmax(np.linalg.norm(Kb-Kb.mean(0),axis=1)))]])
print("nb de clusters utilises par ref :", len(np.unique(lab_ref)))
