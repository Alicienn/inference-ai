# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.endswith("but the sink-plus-background shape does not."):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
lines[i0+1:i0+1] = [
 "",
 "At longer context the shape gains a third component. Repeating the decomposition at",
 "$T=2048$ the first decile still holds $0.63$ of the out-of-window mass and $0.92$ of",
 "the argmaxes remain in the first ten positions, but the remaining deciles are no",
 "longer flat: their share rises monotonically from $0.019$ at the third decile to",
 "$0.082$ at the last one---a recency gradient peaking immediately behind the window.",
 "The boundary gap therefore sits in the densest part of the background, which is the",
 "worst place to lose keys, while effective support grows from $15$ keys at $T=512$ to",
 "$59$ at $T=2048$ and the sink's share falls from $0.71$ to $0.63$. The measured",
 "penalty nevertheless stays small ($0.004$\\,nat at $T=2048$ in the scaled",
 "configuration), so the gradient shifts the mechanism without overturning it.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok :", len("\n".join(lines)), "octets")
