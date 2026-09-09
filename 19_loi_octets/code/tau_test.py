# -*- coding: utf-8 -*-
"""Verification de la constante de decroissance : sigma = C 2^-k => ln sigma = ln C - b/tau,
tau = L D/(8 ln 2) = 738.7 o. On mesure tau sur les plages ou l'erreur est asymptotique."""
import json, pathlib, numpy as np
R = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
fr = json.loads((R / "frontiere.json").read_text(encoding="utf-8"))
TAU = 64 * 64 / (8 * np.log(2))
print(f"tau predit = L D/(8 ln2) = {TAU:.1f} o\n")
print(f"{'reglage':16s} {'plage':>16s} {'tau mesure':>11s} {'tau/predit':>11s}")
for lbl, f in fr.items():
    if "quant 4b" not in f or len(f) < 30: continue
    q = sorted([(v["bytes"], v["sigma_disc"], k) for k, v in f.items()
                if k.startswith("quant ") and v["sigma_disc"] > 0], key=lambda t: t[0])
    for kmin in (1, 3, 4):
        sel = [t for t in q if int(t[2].split()[1].rstrip("b")) >= kmin]
        if len(sel) < 2: continue
        b = np.array([t[0] for t in sel]); s = np.array([t[1] for t in sel])
        a, _ = np.polyfit(b, np.log(s), 1)
        tau = -1.0 / a
        print(f"{lbl:16s} {'k>='+str(kmin)+' ('+str(int(b[0]))+'-'+str(int(b[-1]))+' o)':>16s} "
              f"{tau:11.1f} {tau/TAU:11.3f}")
    # pente log-log locale vs b/tau predit
    b = np.array([t[0] for t in q]); s = np.array([t[1] for t in q])
    loc = np.diff(np.log(s)) / np.diff(np.log(b)); bm = 0.5 * (b[1:] + b[:-1])
    print("    pente locale mesuree :", " ".join(f"{x:.2f}" for x in loc))
    print("    b/tau predit          :", " ".join(f"{x/TAU:.2f}" for x in bm))
