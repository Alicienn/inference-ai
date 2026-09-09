# -*- coding: utf-8 -*-
"""Meme banc sur Qwen3-8B (H=14, KVH=2)."""
import sys, json, pathlib, numpy as np
sys.path.insert(0, r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code")
import err_attention2 as E
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats")
d = np.load(P / "qk_qwen8k.npz", mmap_mode="r")
meta = json.loads(str(d["meta"])); H, KVH = meta["H"], meta["KVH"]
print(f"qwen8k : H={H} KVH={KVH} D={meta['D']} layers={meta['layers']}")
out = []
for W in (512, 0):
    for layer in meta["layers"][:4]:
        for head in (0, H // 2):
            r = E.run("qwen8k", 0, layer, head, head // (H // KVH), W=W)
            r["W"] = W
            out.append(r)
            print(f"  W={W:3d} L{r['layer']:2d} h{r['head']:2d} | masse loc={r['masse_locale']:.3f} "
                  f"oracle={r['masse_sel_oracle']:.3f} passe1={r['masse_passe1']:.3f} | "
                  f"err sel={r['sel_oracle']:.4f} (sans loc {r.get('sel_oracle_noloc', float('nan')):.4f}) | "
                  f"quant " + " ".join(f"{b}b:{r['quant_full_'+str(b)]:.4f}" for b in (2,4,6,8)) + " | "
                  f"ASP " + " ".join(f"{b}b:{r['asp_'+str(b)]:.4f}" for b in (2,4,6,8)))
(OUT / "err_attention_qwen_v2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("->", OUT / "err_attention_qwen_v2.json")
