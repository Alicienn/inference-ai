# -*- coding: utf-8 -*-
"""Diagnostic : la masse d'attention depend-elle de la position dans le bloc ?
Si les positions r=0 (mod 64) recoivent systematiquement plus de masse, le selecteur par
cle a l'offset 0 exploite une regularite POSITIONNELLE, pas la structure du contenu."""
import numpy as np, json, pathlib
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")


def sma(S):
    S = S - S.max(axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=1, keepdims=True)


def diag(tag, doc=0, Lb=64, W=512, nq=8):
    d = np.load(P / f"qk_{tag}.npz", mmap_mode="r")
    meta = json.loads(str(d["meta"]))
    H, KVH, D, layers = meta["H"], meta["KVH"], meta["D"], meta["layers"]
    grp = H // KVH
    mass_r = np.zeros(Lb); cnt = 0
    norm_r = np.zeros(Lb); cntn = 0
    for layer in layers:
        K = np.asarray(d[f"{doc}_{layer}_k_rope"]).astype(np.float64)
        Q = np.asarray(d[f"{doc}_{layer}_q_rope"]).astype(np.float64)
        T = K.shape[0]
        for p in range(T - nq, T):
            for kv in range(KVH):
                qg = Q[p, kv * grp:(kv + 1) * grp]
                Kc = K[:p + 1, kv]
                S = (qg @ Kc.T) / np.sqrt(D)
                P_ = sma(S)
                nbc = max(0, (p - W + 1) // Lb)
                if nbc < 8:
                    continue
                Mb = P_[:, :nbc * Lb].reshape(len(qg), nbc, Lb).sum((0, 1))   # (Lb,)
                mass_r += Mb; cnt += 1
                Nm = np.linalg.norm(Kc[:nbc * Lb], axis=1).reshape(nbc, Lb).mean(0)
                norm_r += Nm; cntn += 1
    mass_r /= max(cnt, 1); norm_r /= max(cntn, 1)
    return mass_r, norm_r


if __name__ == "__main__":
    for tag in ("smol8k", "qwen8k"):
        mr, nr = diag(tag)
        tot = mr.sum()
        order = np.argsort(-mr)
        print(f"\n{tag} : masse totale (region candidate) = {tot:.2f} ; part de l'offset 0 = {100*mr[0]/tot:.2f} %")
        print(f"   top-6 offsets par masse : " + ", ".join(f"r={r} ({100*mr[r]/tot:.2f} %)" for r in order[:6]))
        print(f"   offsets les plus faibles: " + ", ".join(f"r={r} ({100*mr[r]/tot:.2f} %)" for r in order[-3:]))
        print(f"   masse r=0 {mr[0]:.2f} vs moyenne {tot/64:.2f} -> rapport {mr[0]/(tot/64):.2f}")
        print(f"   norme de cle : r=0 {nr[0]:.2f} vs moyenne {nr.mean():.2f} -> rapport {nr[0]/nr.mean():.2f}")
        print(f"   correlation masse/norme par offset : {np.corrcoef(mr, nr)[0,1]:.3f}")
