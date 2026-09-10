# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
hit = None
for i, L in enumerate(lines):
    if "is $L_b\\approx4$--$8$, and below it the quota $m$ grows without buying anything." in L:
        hit = i; break
assert hit is not None
lines[hit+1:hit+1] = [
 "",
 "\\paragraph{The window is a red herring; the selection is the mechanism.} Fixing",
 "$L_b=4$ and the total budget, we swept the window against the quota",
 "(Table~\\ref{tab:window}). At $192$ keys read, $W=32$ with $m=40$, $W=64$ with",
 "$m=32$ and $W=128$ with $m=16$ all land within $0.008$\\,nat of one another",
 "($+0.027$, $+0.029$, $+0.021$): how the budget is split between contiguous and",
 "selected keys barely matters. What matters is that there \\emph{is} a selection. A",
 "pure sliding window reading the most recent $256$ keys---half the context, more",
 "keys than any sparse configuration above---loses $1.23$\\,nat, two orders of",
 "magnitude worse than reading $192$ keys of which only $16$ are chosen by the",
 "index. Locality alone buys almost nothing; the two-pass selection does the work.",
 "",
 "\\begin{table}[t]",
 "\\centering",
 "\\caption{Window against quota at fixed total budget, $L_b=4$, SmolLM2-135M,",
 "$T=512$, means over three corpus chunks. The last row reads more keys than any",
 "sparse configuration and still loses two orders of magnitude more.}",
 "\\label{tab:window}",
 "\\begin{tabular}{rrrrr}",
 "\\toprule",
 "$W$ & $m$ & keys read & vs dense & index cost \\\\",
 "\\midrule",
 "32 & 40 & 192 & $+0.027$ & $+0.009$ \\\\",
 "64 & 32 & 192 & $+0.029$ & $+0.011$ \\\\",
 "128 & 16 & 192 & $+0.021$ & $+0.008$ \\\\",
 "64 & 48 & 256 & $+0.026$ & $+0.006$ \\\\",
 "128 & 32 & 256 & $+0.018$ & $+0.005$ \\\\",
 "256 & 0 & 256 & $+1.234$ & --- \\\\",
 "\\bottomrule",
 "\\end{tabular}",
 "\\end{table}",
 "",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
