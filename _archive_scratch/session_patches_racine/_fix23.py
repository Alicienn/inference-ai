# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
hit = None
for i, L in enumerate(lines):
    if "behind the training-free PCA basis; this last comparison is the one" in L:
        hit = i; break
assert hit is not None
assert "still being re-measured." in lines[hit + 1], lines[hit + 1]
lines[hit] = "$0.019$\\,nat behind the training-free PCA basis at the same dimension"
lines[hit + 1] = ("($d'=4$), while sitting $0.035$\\,nat \\emph{ahead} of it in sample: the gap is")
lines[hit + 2:hit + 2] = ["overfitting, not capacity."]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
