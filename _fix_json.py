import json, pathlib
R = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
for L in (32, 128):
    p = R / f"loi_L{L}_s0.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["quant"] = {k: {"bytes": v["bytes"], "sigma_disc": v.get("sigma_disc", v.get("sigma")), "r@8": v.get("r@8", 0.0)}
                  for k, v in d["quant"].items()}
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"{p.name}: {len(d['quant'])} points, cles = {sorted(d['quant'][list(d['quant'])[0]].keys())}")
