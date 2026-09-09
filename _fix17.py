# -*- coding: utf-8 -*-
import pathlib, re
D = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code")
fixed = []
for f in sorted(D.glob("*.py")):
    t = f.read_text(encoding="utf-8")
    if "ti.clamp(min=0), True)" not in t:
        continue
    out = []
    for line in t.split("\n"):
        m = re.match(r"^(\s*)onehot\.scatter_\(2, ti\.clamp\(min=0\), True\)\s*$", line)
        if m:
            ind = m.group(1)
            out.append(f"{ind}_vv = ti >= 0")
            out.append(f"{ind}_hh, _pp, _ = torch.nonzero(_vv, as_tuple=True)")
            out.append(f"{ind}onehot[_hh, _pp, ti[_vv]] = True")
            fixed.append(f.name)
        else:
            out.append(line)
    f.write_text("\n".join(out), encoding="utf-8")
print("fichiers corriges :", len(fixed))
for n in fixed:
    print("  ", n)

# script de remesure Qwen (copie du harnais corrige)
src = (D / "courbe_corrigee.py").read_text(encoding="utf-8")
src = src.replace('MID = "HuggingFaceTB/SmolLM2-135M"', 'MID = "Qwen/Qwen2.5-0.5B"')
src = src.replace("OFFSETS = [0, 200000, 400000]", "OFFSETS = [0, 200000]")
src = src.replace("courbe_corrigee3.txt", "remesure_qwen.txt")
src = src.replace('for k in ["dense", "maxip", "oracle", "pca_d16", "pca_d8", "aleatoire"]',
                  'for k in ["dense", "maxip", "oracle", "pca_d16", "pca_d8", "aleatoire"]')
(D / "remesure_qwen.py").write_text(src, encoding="utf-8")
print("remesure_qwen.py ecrit")
