# -*- coding: utf-8 -*-
"""Test de mise a l'echelle en la longueur de bloc L.
Prediction : sigma = C 2^(-b/(L D/8)) => tau = L D/(8 ln2) et sigma(4b)/sigma(8b) = 16 pour tout L.
"""
import sys, json, pathlib, numpy as np
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\12_poc\code")
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code")
import frontiere2 as F2
Lb = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
res, rad = F2.run("qwen8k", Lb=Lb, nq=10, use_rope=True, seed=seed, max_cfg=4, hstride=4)
q = sorted([(v["bytes"], v["sigma_disc"], k) for k, v in res.items() if k.startswith("quant ")], key=lambda t: t[0])
out = {"L": Lb, "seed": seed, "quant": {k: {"bytes": float(b), "sigma": float(s)} for b, s, k in q}}
for b, s, k in q:
    print(f"  {k:8s} {b:7.0f} o  sigma={s:.5f}")
sel = [t for t in q if int(t[2].split()[1].rstrip("b")) >= 4]
b = np.array([t[0] for t in sel]); s = np.array([t[1] for t in sel])
a, _ = np.polyfit(b, np.log(s), 1)
out["tau_measured"] = float(-1/a)
out["tau_predicted"] = float(Lb * 64 / (8 * np.log(2)))
out["ratio_4b_8b"] = float(sel[0][1] / sel[-1][1])
print(f"  tau mesure = {out['tau_measured']:.1f} o  |  tau predit = {out['tau_predicted']:.1f} o  "
      f"|  ratio = {out['tau_measured']/out['tau_predicted']:.3f}")
print(f"  sigma(4b)/sigma(8b) = {out['ratio_4b_8b']:.2f}  (predit 16.00 pour tout L)")
p = pathlib.Path(rf"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats\loi_L{Lb}_s{seed}.json")
p.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("  ecrit:", p.name)
