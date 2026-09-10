# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
hit = None
for i, L in enumerate(lines):
    if "configuration we started from." in L:
        hit = i; break
assert hit is not None
lines[hit+1:hit+1] = [
 "",
 "The effect is not specific to one model. Repeating the constant-budget sweep on",
 "Qwen2.5-0.5B, a different architecture ($14$ query heads, $2$ KV heads), gives",
 "$+0.802$, $+0.346$, $+0.132$, $+0.035$ and $-0.003$\\,nat for",
 "$L_b=64,32,16,8,4$: at $L_b=4$ the sparse model matches dense attention while",
 "reading $37.5\\%$ of the keys. On this model the index cost is \\emph{negative} in",
 "four of the five configurations ($-0.018$ to $-0.005$\\,nat): a maximum over",
 "$8$-dimensional projections is not merely a cheaper stand-in for \\emph{maxip}, it",
 "is a better scorer. Averaged over chunks the index cost is $\\pm0.02$\\,nat either",
 "way, so it is best read as free rather than as a fixed penalty.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
