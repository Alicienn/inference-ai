# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
t = p.read_text(encoding="utf-8")
old = "to chance. Distilling the basis"
new = ("to chance---numbers measured under the earlier harness and being re-measured\n"
       "with the corrected one, as are the scaling-law and learned-index comparisons. Distilling the basis")
assert old in t
t = t.replace(old, new, 1)
p.write_text(t, encoding="utf-8")
print("mise en garde ajoutee |", len(t), "octets")
