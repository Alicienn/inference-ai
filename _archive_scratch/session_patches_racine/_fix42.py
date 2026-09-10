# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.startswith("\\paragraph{What sets $\\rho$.}"):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
i1 = i0
while not lines[i1].endswith("measurable but not derivable from the window mass alone."):
    i1 += 1
lines[i0:i1 + 1] = [
 "\\paragraph{What sets $\\rho$: a sink plus a uniform background.} A natural guess is",
 "that attention outside the window is spread uniformly, giving",
 "$\\rho=(1-\\mu_w)/(T-W)$. The measured density just behind the window is $2$--$5\\times$",
 "below the global average of non-window keys, which first looks like a trough.",
 "Decomposing the out-of-window distribution resolves it: that distribution has an",
 "effective support of only $15$ keys, and a single key holds $0.67$ of it---the",
 "sequence-initial sink, which sits in the first tenth of the positions and carries",
 "$0.70$ of the out-of-window mass. The remaining $0.30$ is a flat background at",
 "$8$--$9\\times10^{-4}$ per key. So the background \\emph{is} uniform and $\\rho$ measures",
 "it; the apparent trough was the background compared against an average inflated by",
 "the sink. This also explains why the granularity penalty is small and why block",
 "selection works at all: the sink lives in the oldest block, which is always eligible",
 "and therefore never trapped in the boundary gap, so the gap only ever takes",
 "background mass---and the background is flat, which is exactly why ranking it is a",
 "near-tie problem. At $W=128$ and $T=512$ the out-of-window mass is $0.58$ of the",
 "total, so this is not a marginal component.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
