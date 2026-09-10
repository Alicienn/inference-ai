# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
hit = None
for i, L in enumerate(lines):
    if "for the selector to stay within a few hundredths of dense attention." in L:
        hit = i; break
assert hit is not None
lines[hit+1:hit+1] = [
 "",
 "\\paragraph{A third architecture, without rotary embeddings.} GPT-2 has no rotary",
 "embeddings, uses learned absolute positions, and projects queries, keys and values",
 "with a single matrix, so nothing in the mechanism above is specific to the",
 "Llama-style stack. The results carry over. At a fixed $192$ keys the cost against",
 "dense falls from $0.400$\\,nat at $L_b=64$ to $0.216$ at $L_b=32$ and $-0.004$ at",
 "$L_b=4$; at $128$ keys the selector costs $0.020$\\,nat while a pure window of the",
 "same size costs $1.42$\\,nat, a factor of $70$. The PCA index again beats",
 "\\emph{maxip} in every configuration ($-0.008$ to $-0.017$\\,nat).",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
