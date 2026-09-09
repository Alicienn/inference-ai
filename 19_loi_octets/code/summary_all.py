import json, pathlib, numpy as np
R = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
fr = json.loads((R / "frontiere.json").read_text(encoding="utf-8"))
pa = json.loads((R / "asp_pareto.json").read_text(encoding="utf-8"))

def fit(f, pref):
    pts = sorted([(v["bytes"], v["sigma_disc"]) for k, v in f.items() if k.startswith(pref) and v["sigma_disc"] > 0])
    if len(pts) < 2: return None
    a, b = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)
    return -a

print("=== FRONTIERE : exposants et valeurs cles par reglage ===")
print(f"{'reglage':18s} {'a_coreset':>10s} {'a_quant':>8s} {'sig(cs r16)':>11s} {'sig(q4b)':>9s} {'sig(q6b)':>9s} {'sig(q2b)':>9s} {'gap?':>6s}")
for lbl, f in fr.items():
    if "coreset r=16" not in f: continue
    ac, aq = fit(f, "coreset r="), fit(f, "quant ")
    print(f"{lbl:18s} {ac:10.3f} {aq:8.3f} {f['coreset r=16']['sigma_disc']:11.4f} "
          f"{f['quant 4b']['sigma_disc']:9.4f} {f['quant 6b']['sigma_disc']:9.4f} {f['quant 2b']['sigma_disc']:9.4f} "
          f"{'OUI' if f['quant 6b']['sigma_disc'] < 0.07 else 'non':>6s}")
    print(f"   octets : coreset r=16 {f['coreset r=16']['bytes']:.0f} | quant 4b {f['quant 4b']['bytes']:.0f} | "
          f"quant 6b {f['quant 6b']['bytes']:.0f} | bloc brut {f['coreset r=64']['bytes']:.0f} ; "
          f"sigma coreset r=64 = {f['coreset r=64']['sigma_disc']:.2e}")

print("\n=== PARETO : points de frontiere par reglage ===")
for lbl, v in pa.items():
    res = v["asp"]
    front = []
    for k, vv in sorted(res.items(), key=lambda kv: kv[1]["octets"]):
        if not front or vv["rappel"] > front[-1][2] + 1e-9:
            front.append((k, vv["octets"], vv["rappel"]))
    hi = [t for t in front if t[2] >= 0.90]
    print(f"  {lbl}: {len(front)} points, {len(hi)} a >=90% ; "
          f"tous quantifies au-dessus de 90% : {all('quant' in k for k,_,_ in hi)}")
    for k, b, r in front[:6]:
        print(f"     {b:10.0f} o {100*r:6.2f}%  {k}")
    print("     ...")
