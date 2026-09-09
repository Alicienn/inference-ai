import sys, numpy as np
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code")
import frontiere2 as F2
res, rad = F2.run("qwen8k", Lb=64, nq=6, use_rope=True, max_cfg=2, hstride=3)
for k in ("coreset r=1", "coreset r=8", "coreset r=16", "cobs r=16", "quant 2b", "quant 4b"):
    v = res[k]
    print(f"  {k:14s} {v['bytes']:7.0f} o  sigma_disc={v['sigma_disc']:.4f}  r@8={100*v['r@8']:6.2f}%")
dmed = np.median(rad, 0); rr = np.arange(1, len(dmed) + 1)
sel = (rr >= 2) & (rr <= 16) & (dmed > 0)
a, b = np.polyfit(np.log(rr[sel]), np.log(dmed[sel]), 1)
print(f"d_cov = {-1/a:.2f}   (slope {a:.3f})")
print("assert quant4b<coreset16:", res["quant 4b"]["sigma_disc"] < res["coreset r=16"]["sigma_disc"])
print("assert d_cov in (3,7):", 3.0 < -1/a < 7.0)
