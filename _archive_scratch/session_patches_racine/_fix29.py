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
 "\\paragraph{Granularity, not the number of blocks, is the dominant knob.} Our",
 "headline configuration ($L_b=32$, $m=2$, window $128$) was one point in a grid we",
 "had not swept. We swept it: $L_b\\in\\{16,32,64\\}$, $m\\in\\{1,2,4\\}$, three corpus",
 "chunks, with the index and the baselines recomputed for every configuration",
 "(Table~\\ref{tab:sweep}). Finer blocks are strictly better at equal or lower cost.",
 "At a byte fraction of $31\\%$, $L_b=16$ with $m=2$ costs $0.123$\\,nat against the",
 "dense model, while $L_b=32$ with $m=1$ costs $0.245$\\,nat; and $L_b=16$ with a",
 "single block reads $28\\%$ of the keys and still beats $L_b=32$ with two blocks",
 "($0.153$ against $0.214$\\,nat). Coarse blocks are far worse: $L_b=64$ costs",
 "$0.36$--$0.39$\\,nat even when it reads $75\\%$ of the keys. The mechanism is a",
 "boundary effect. With a window of $W$, a block grid of size $L_b$ leaves a",
 "systematic gap of up to $L_b-1$ keys between the window and the nearest eligible",
 "block, so keys just outside the window---often the most relevant distant",
 "ones---are dropped, while the coarse blocks that are admitted bring in irrelevant",
 "keys. The index cost itself shrinks with $L_b$ and $m$ ($+0.058$ down to",
 "$+0.001$\\,nat) and is sometimes \\emph{negative}: at $m=4$ the PCA index",
 "occasionally beats \\emph{maxip}. Granularity is therefore a cheaper knob than",
 "index accuracy, and $L_b$ should be made as small as the scorer can still rank.",
 "",
 "\\begin{table}[t]",
 "\\centering",
 "\\caption{Sweep over block size $L_b$ and quota $m$ (SmolLM2-135M, $T=512$,",
 "window $128$, causal PCA index with $d'=8$; means over three corpus chunks).",
 "``keys'' is the fraction $(mL_b+W)/T$ of key positions read; ``index cost'' is",
 "the loss of the PCA index against \\emph{maxip} at the same configuration.}",
 "\\label{tab:sweep}",
 "\\begin{tabular}{rrrrr}",
 "\\toprule",
 "$L_b$ & $m$ & keys & vs dense & index cost \\\\",
 "\\midrule",
 "16 & 1 & $28.1\\%$ & $-0.153$ & $+0.058$ \\\\",
 "16 & 2 & $31.2\\%$ & $-0.123$ & $+0.026$ \\\\",
 "16 & 4 & $37.5\\%$ & $-0.114$ & $+0.007$ \\\\",
 "32 & 1 & $31.2\\%$ & $-0.245$ & $+0.056$ \\\\",
 "32 & 2 & $37.5\\%$ & $-0.214$ & $+0.011$ \\\\",
 "32 & 4 & $50.0\\%$ & $-0.206$ & $+0.004$ \\\\",
 "64 & 1 & $37.5\\%$ & $-0.385$ & $+0.026$ \\\\",
 "64 & 2 & $50.0\\%$ & $-0.363$ & $+0.007$ \\\\",
 "64 & 4 & $75.0\\%$ & $-0.360$ & $+0.001$ \\\\",
 "\\bottomrule",
 "\\end{tabular}",
 "\\end{table}",
 "",
]
lines[hit:hit] = new
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
