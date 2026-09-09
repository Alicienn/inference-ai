import json, pathlib
R = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
fr = json.loads((R / "frontiere.json").read_text(encoding="utf-8"))
for lbl, f in fr.items():
    ks = sorted(f.keys())
    print(f"{lbl}: {len(ks)} familles ; quantG present = {any('quantG' in k for k in ks)} ; "
          f"quant1b = {any('quant 1b' == k for k in ks)}")
    print("   ", ks[:8], "...")
import os, time
for p in sorted(R.glob('*.json')):
    print(f"  {p.name:24s} {p.stat().st_size:8d} o  {time.strftime('%H:%M:%S', time.localtime(p.stat().st_mtime))}")
