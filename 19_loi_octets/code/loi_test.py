# -*- coding: utf-8 -*-
"""Loi de puissance veritable ou exponentielle locale ?
sigma = C b^-alpha  (pente log-log constante)  vs  sigma = C 2^{-b/tau} (pente locale ~ b/tau).
Prediction : tau = L*D/8 = 512 octets (car k = 8(b - 4D)/(L D) et l'erreur ~ 2^-k).
"""
import json, pathlib, numpy as np
R = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
fr = json.loads((R / "frontiere.json").read_text(encoding="utf-8"))
LD8 = 64 * 64 / 8.0

def pts(f, pref):
    return sorted([(v["bytes"], v["sigma_disc"]) for k, v in f.items()
                   if k.startswith(pref) and v["sigma_disc"] > 0])

for lbl, f in fr.items():
    if "quant 4b" not in f: continue
    print("=" * 88)
    print(f"{lbl}   (tau predit = L*D/8 = {LD8:.0f} o)")
    for pref in ("quant ", "quantG ", "coreset r=", "cobs r="):
        P = pts(f, pref)
        if len(P) < 3: continue
        b = np.array([p[0] for p in P]); s = np.array([p[1] for p in P])
        # 1) loi de puissance globale (log-log)
        a1, _ = np.polyfit(np.log(b), np.log(s), 1)
        r2_pow = 1 - np.sum((np.log(s) - np.polyval(np.polyfit(np.log(b), np.log(s), 1), np.log(b)))**2) / np.sum((np.log(s) - np.log(s).mean())**2)
        # 2) loi exponentielle (ln sigma vs b)
        a2, b2 = np.polyfit(b, np.log(s), 1)
        tau = -1.0 / a2
        pred = np.polyval([a2, b2], b)
        r2_exp = 1 - np.sum((np.log(s) - pred)**2) / np.sum((np.log(s) - np.log(s).mean())**2)
        # 3) pentes log-log LOCALES (entre points successifs)
        loc = np.diff(np.log(s)) / np.diff(np.log(b))
        bmid = 0.5 * (b[1:] + b[:-1])
        print(f"  {pref:12s} n={len(P):2d}  alpha_global={-a1:6.3f} (R2={r2_pow:.4f})  "
              f"tau={tau:7.1f} o (R2_exp={r2_exp:.4f})")
        print(f"      pente locale : " + "  ".join(f"{bb:.0f}o:{ll:5.2f}" for bb, ll in zip(bmid, loc)))
        print(f"      b/tau_predit   : " + "  ".join(f"{bb:.0f}o:{bb/LD8:5.2f}" for bb in bmid))
