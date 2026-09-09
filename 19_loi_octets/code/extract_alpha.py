"""Loi octets-precision des resumes de blocs : extraction de l'exposant alpha.

On lit law_selection.json (deja produit) et on ajuste sigma = C * cost^(-alpha)
sur la famille coreset, puis on convertit en octets reels.
"""
import json, pathlib
import numpy as np

RES = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")
d = json.loads((RES / "law_selection.json").read_text(encoding="utf-8"))

print("=== sigma_disc vs cout (1 vecteur = D*2 octets fp16 ; D=64 -> 128 o) ===")
for reg, v in d.items():
    print("==", reg)
    ms = v["methods"]
    for m, mm in sorted(ms.items(), key=lambda kv: kv[1]["cost"]):
        print("  %-14s cost=%6.2f octets=%7.0f  sperp=%7.4f sigma=%7.4f rmse=%7.3f r8=%.4f"
              % (m, mm["cost"], mm["cost"] * 128, mm.get("sperp", float("nan")),
                 mm.get("sigma", float("nan")), mm["rmse"], mm.get("r@8", float("nan"))))
    pts = [(mm["cost"], mm.get("sperp", mm.get("sigma")))
           for m, mm in ms.items() if m.startswith("coreset")]
    pts = sorted([(c, s) for c, s in pts if s and s > 0])
    if len(pts) > 2:
        x = np.log(np.array([p[0] for p in pts])); y = np.log(np.array([p[1] for p in pts]))
        a, b = np.polyfit(x, y, 1)
        print("   -> ajustement coreset : sigma = %.4f * cost^(%.3f)   alpha=%.3f  1/alpha=%.2f"
              % (np.exp(b), a, -a, -1.0 / a))
        # exposants locaux
        for i in range(1, len(pts)):
            c0, s0 = pts[i - 1]; c1, s1 = pts[i]
            print("      local r %.0f->%.0f : alpha=%.3f"
                  % (c0, c1, -np.log(s1 / s0) / np.log(c1 / c0)))
