import json, pathlib
R = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
fr = json.loads((R / "frontiere.json").read_text(encoding="utf-8"))
pa = json.loads((R / "asp_pareto.json").read_text(encoding="utf-8"))
print("=== A. frontiere octets-precision, qwen8k|RoPE ===")
f = fr["qwen8k|RoPE"]
for k in ("coreset r=1","coreset r=2","coreset r=8","coreset r=16","coreset r=32","coreset r=64",
          "cobs r=16","quant 1b","quant 2b","quant 3b","quant 4b","quant 6b","quant 8b"):
    v = f[k]
    print(f"  {k:14s} {v['bytes']:7.0f} o  sigma_disc={v['sigma_disc']:.6f}  r@8={100*v['r@8']:6.2f}%")
print("\n=== B. frontiere a deux passes (qwen8k|RoPE) ===")
res = pa["qwen8k|RoPE"]["asp"]
front = []
for k, v in sorted(res.items(), key=lambda kv: kv[1]["octets"]):
    if not front or v["rappel"] > front[-1][2] + 1e-9:
        front.append((k, v["octets"], v["rappel"]))
for k, b, r in front:
    print(f"  {b:9.0f} o  {100*r:6.2f}%  {k}")
print("\n=== C. domination de la configuration du papier (passe 2 = coreset r=16) ===")
items = [(v["octets"], v["rappel"], k) for k, v in res.items()]
for k, v in sorted(res.items(), key=lambda kv: kv[1]["octets"]):
    if "coreset r=16" not in k:
        continue
    dom = [(b, r, nm) for b, r, nm in items if b <= v["octets"] and r >= v["rappel"]+0.005 and nm != k]
    if dom:
        b, r, nm = min(dom)
        print(f"  {k:40s} {v['octets']:9.0f} o {100*v['rappel']:6.2f}%  DOMINE par {nm} ({b:.0f} o, {100*r:.2f}%)")
    else:
        print(f"  {k:40s} {v['octets']:9.0f} o {100*v['rappel']:6.2f}%  (non domine)")
print("\n=== D. plafond passe-2 exacte vs quant 4b ===")
for m in ("2m","4m","8m"):
    c = res[f"coreset r=1 -> exact m1={m}"]; g = res[f"coreset r=1 -> quant 4b m1={m}"]
    print(f"  m1={m}: exact {100*c['rappel']:.2f}% @ {c['octets']:.0f} o | quant4b {100*g['rappel']:.2f}% @ {g['octets']:.0f} o"
          f" | {c['octets']/g['octets']:.2f}x moins d'octets, ecart {100*(c['rappel']-g['rappel']):.2f} pt")
print("\n=== E. autres reglages (frontiere, 3 premiers points et le point quant 4b) ===")
for lbl in pa:
    r = pa[lbl]["asp"]
    fr2 = []
    for k, v in sorted(r.items(), key=lambda kv: kv[1]["octets"]):
        if not fr2 or v["rappel"] > fr2[-1][2] + 1e-9:
            fr2.append((k, v["octets"], v["rappel"]))
    print(f"  {lbl}: {len(fr2)} points ; sommet = {fr2[-1][0]} ({fr2[-1][1]:.0f} o, {100*fr2[-1][2]:.2f}%)")
