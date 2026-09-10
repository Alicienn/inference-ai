# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.endswith("mechanism rather than the system."):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
lines[i0+1:i0+1] = [
 "",
 "\\paragraph{The deployable index beats dense at long context.} The contrasts above use",
 "oracle selection. Replacing it with the deployable causal PCA index---base fitted on",
 "the first $256$ tokens, block score",
 "$\\max_{k\\in B}\\max_{i\\le 8}\\langle q,u_i\\rangle\\langle k,u_i\\rangle$---at the same",
 "fixed read fraction of $37.5\\%$ and $L_b=4$ gives, at $T=1024$: dense $13.5125$,",
 "oracle $13.4733$ ($-0.039$), index $13.4768$ ($-0.036$); at $T=2048$: dense",
 "$13.5443$, oracle $13.5179$ ($-0.026$), index $13.4977$ ($-0.047$). Both beat dense,",
 "and at $T=2048$ the index beats the attention-peak oracle. Two caveats bound this.",
 "First, the sign is not robust to how blocks are scored: selecting by summed mass",
 "instead of peak attention gives $+0.022$ at $T=1024$, so aggregation is a first-order",
 "design choice, not a detail. Second, this is loss at a $2.7\\times$ KV reduction, not",
 "latency; the selector cost that drives the negative headline is untouched by it.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok :", len("\n".join(lines)), "octets")
