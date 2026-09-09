import json, pathlib, numpy as np
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats")
rows = json.loads((OUT / "reconciliation_sink.json").read_text(encoding="utf-8"))
M = ("mean_r1", "mean_r1_sink", "s1_off0", "s1_off16_sink", "oracle")
lines = []
for tag in ("smol8k", "qwen8k"):
    sel = [x for x in rows if x["tag"] == tag]
    heads = {}
    for x in sel:
        heads.setdefault((x["layer"], x["head"]), []).append(x)
    sk = np.mean([np.mean([y["masse_sink"] for y in v]) for v in heads.values()])
    lines.append(f"{tag} : {len(heads)} tetes | masse du bloc du sink = {sk:.4f}")
    for name in M:
        e = np.array([np.mean([y[f"err_{name}"] for y in v]) for v in heads.values()])
        ms = np.array([np.mean([y[f"mass_{name}"] for y in v]) for v in heads.values()])
        lines.append(f"   {name:15s} err med={np.median(e):.4f} p90={np.percentile(e,90):.4f} "
                     f"max={e.max():.4f} | catastrophes={100*(e>0.5).mean():5.1f} % | masse med={np.median(ms):.3f}")
    lines.append("")
(OUT / "reconciliation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("ok")
