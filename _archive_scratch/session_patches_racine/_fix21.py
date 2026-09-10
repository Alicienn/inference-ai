# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
done = False
for i, L in enumerate(lines):
    if "chance. Distilling the basis" in L:
        lines[i] = L.replace("chance. Distilling the basis",
            "chance---numbers measured under the earlier harness and being re-measured with the\n"
            "corrected one, as are the scaling-law and learned-index comparisons. Distilling the basis")
        done = True; break
print("insere =", done)
if done:
    p.write_text("\n".join(lines), encoding="utf-8")
