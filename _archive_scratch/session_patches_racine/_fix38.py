# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
hit = None
for i, L in enumerate(lines):
    if "\\emph{maxip} in every configuration ($-0.008$ to $-0.017$\\,nat)." in L:
        hit = i; break
assert hit is not None
lines[hit+1:hit+1] = [
 "",
 "\\paragraph{Scaling the configuration with the context.} The scale law above held",
 "the configuration fixed and let the fraction of keys read shrink. Scaling the",
 "configuration with the context instead---keeping the fraction at a quarter by",
 "setting $W=T/8$ and $m=T/32$ blocks of $L_b=4$---the cost against dense",
 "\\emph{falls}: $+0.041$\\,nat at $T=512$, $+0.012$ at $1024$ and $+0.004$ at",
 "$2048$. At the same fraction a pure window costs $+1.76$, $+1.24$ and $+1.08$\\,nat,",
 "so the ratio between selection and locality grows from $43\\times$ to $299\\times$.",
 "The advantage of selection over locality widens with context length, the opposite",
 "of what a utility-threshold argument based on fixed overhead would predict.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
