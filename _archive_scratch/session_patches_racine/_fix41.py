# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.endswith("context grows, so the optimal $L_b$ stays small and the penalty shrinks."):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
lines[i0+1:i0+1] = [
 "",
 "\\paragraph{What sets $\\rho$.} A natural guess is that attention outside the window is",
 "spread uniformly, giving $\\rho=(1-\\mu_w)/(T-W)$ with $\\mu_w$ the window mass. The",
 "measurement refutes it. Over a grid of $(T,W)$ the density immediately behind the",
 "window is $0.18$ to $0.49$ times the global average density of non-window keys: there",
 "is a \\emph{trough} just behind the window, not a uniform floor. The trough ratio",
 "depends mainly on $W/T$ ($0.49$ at $W/T=1/16$, $0.18$ at $W/T=1/2$), $\\rho$ still",
 "falls roughly as $1/(T-W)$, which is why the penalty shrinks with context, and the",
 "window mass itself grows slowly with $W/T$ ($0.34$ to $0.51$). So $\\rho$ is",
 "measurable but not derivable from the window mass alone.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
