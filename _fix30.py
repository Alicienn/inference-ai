# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
hit = None
for i, L in enumerate(lines):
    if "\\paragraph{What the corrected measurements say.}" in L:
        hit = i; break
assert hit is not None
new = [
 "The boundary effect makes a sharp prediction: holding the byte budget fixed and",
 "shrinking the grid should recover quality without spending a byte. It does. With",
 "$m L_b=64$ and $W=128$---so $37.5\\%$ of the key positions in every",
 "configuration---the cost against the dense model falls from $0.385$\\,nat at",
 "$L_b=64$ to $0.021$\\,nat at $L_b=4$, a factor of $18$, while the index cost stays",
 "flat at $0.007$--$0.011$\\,nat (Table~\\ref{tab:budget}). The growing candidate pool",
 "does not hurt the index: a finer grid also makes the per-block maximum a tighter",
 "proxy for the true maximum, and that gain outweighs having more blocks to rank. At",
 "$L_b=4$ and $m=16$ the sparse model reads $37.5\\%$ of the keys and loses",
 "$0.021$\\,nat to dense attention---an order of magnitude less than the",
 "configuration we started from.",
 "",
 "\\begin{table}[t]",
 "\\centering",
 "\\caption{Constant byte budget ($mL_b=64$, $W=128$, so $37.5\\%$ of key positions",
 "read in every row), SmolLM2-135M, $T=512$, means over three corpus chunks.}",
 "\\label{tab:budget}",
 "\\begin{tabular}{rrrrr}",
 "\\toprule",
 "$L_b$ & $m$ & candidates & vs dense & index cost \\\\",
 "\\midrule",
 "64 & 1 & 8 & $-0.385$ & $+0.026$ \\\\",
 "32 & 2 & 16 & $-0.215$ & $+0.011$ \\\\",
 "16 & 4 & 32 & $-0.115$ & $+0.007$ \\\\",
 "8 & 8 & 64 & $-0.059$ & $+0.008$ \\\\",
 "4 & 16 & 128 & $-0.021$ & $+0.008$ \\\\",
 "\\bottomrule",
 "\\end{tabular}",
 "\\end{table}",
 "",
]
lines[hit:hit] = new
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
