# -*- coding: utf-8 -*-
"""Test decisif : alpha est-il constant (vraie loi de puissance) ou un artefact de fenetre ?
Prediction exponentielle : sigma(k) = C 2^-k, donc rapport par bit -> 2 et alpha(ajuste sur
une fenetre) croit avec la fenetre. Prediction loi de puissance : alpha constant.
tau exact attendu : ln sigma = ln C - b/(L D/(8 ln 2)) -> tau = L D/(8 ln2) = 738.6 o.
"""
import json, pathlib, numpy as np
R = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
fr = json.loads((R / "frontiere.json").read_text(encoding="utf-8"))
tau_pred = 64 * 64 / (8 * np.log(2))
print(f"tau attendu si sigma = C 2^-k : L D/(8 ln2) = {tau_pred:.1f} o\n")

def get(f, pref):
    d = {}
    for k, v in f.items():
        if k.startswith(pref) and v["sigma_disc"] > 0:
            d[k] = (v["bytes"], v["sigma_disc"])
    return d

for lbl, f in fr.items():
    if "quant 4b" not in f: continue
    print("=" * 86)
    print(lbl)
    q = get(f, "quant ")
    ks = sorted(q.items(), key=lambda kv: kv[1][0])
    print("  quant (echelles par bloc) :")
    prev = None
    for k, (b, s) in ks:
        extra = "" if prev is None else f"  rapport={prev[1]/s:5.2f}  (bits {prev[0]}->{k[3:]})"
        print(f"    {k:8s} {b:6.0f} o  sigma={s:.5f}{extra}")
        prev = (k, s)
    print("  alpha ajuste sur fenetres glissantes de 3 points consecutifs :")
    for i in range(len(ks) - 2):
        w = ks[i:i+3]
        bb = np.array([x[1][0] for x in w]); ss = np.array([x[1][1] for x in w])
        a, _ = np.polyfit(np.log(bb), np.log(ss), 1)
        print(f"    {w[0][0]:7s}..{w[-1][0]:7s} ({bb[0]:.0f}-{bb[-1]:.0f} o)  alpha={-a:6.3f}")
    c = get(f, "coreset r=")
    cs = sorted(c.items(), key=lambda kv: kv[1][0])
    print("  coreset : alpha sur les memes tailles de fenetre (3 points consecutifs) :")
    for i in range(len(cs) - 2):
        w = cs[i:i+3]
        bb = np.array([x[1][0] for x in w]); ss = np.array([x[1][1] for x in w])
        a, _ = np.polyfit(np.log(bb), np.log(ss), 1)
        print(f"    {w[0][0]:12s}..{w[-1][0]:12s} ({bb[0]:.0f}-{bb[-1]:.0f} o)  alpha={-a:6.3f}")
