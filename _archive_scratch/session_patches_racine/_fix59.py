# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.endswith("component."):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
lines[i0+1:i0+1] = [
 "",
 "\\paragraph{A valid retrieval bench: GPT-2.} The degeneracy is not universal. On GPT-2",
 "the same repeated-span probe engages a genuine retrieval circuit: the loss on the",
 "fourth copy is $2.95$\\,nat below the first, the attention on the distant source copy",
 "is $0.437$ against $0.250$ uniform, and a single head at layer 11 puts its entire mass",
 "on it. On this bench we can finally ask whether the selector keeps what matters. At a",
 "$37.5\\%$ read budget the oracle keeps the block containing the attention peak in",
 "$99.7\\%$ of (layer, head, query) triples and costs $0.007$\\,nat, while the deployable",
 "index keeps it in $94.4\\%$ and costs $0.226$---a $32\\times$ larger penalty from a",
 "$5.3$-point difference. The loss is steeply non-linear in the peak-miss rate,",
 "$\\Delta\\mathrm{loss}\\approx 4\\times(1-\\text{peak rate})$ in the mid-range, so the",
 "relevant design target is not mass recall but \\emph{peak retention}. Spending budget",
 "buys it: peak retention rises through $89.6\\%$, $94.4\\%$, $96.9\\%$ and $99.5\\%$ at",
 "$25\\%$, $37.5\\%$, $50\\%$ and $75\\%$ of keys, and at $75\\%$ the loss matches dense",
 "($-0.001$\\,nat). Raising the projection rank does not: at $37.5\\%$, $d'=8,16,32$ give",
 "peak retention $94.4\\%$, $94.7\\%$ and $95.1\\%$. The index cannot buy retrieval",
 "fidelity with dimensions, only with bytes. On retrieval tasks ASP therefore saves about",
 "$1.3\\times$, not the $16\\times$ it saves on flowing text---the two regimes sit at",
 "opposite ends of the budget axis.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok :", len("\n".join(lines)), "octets")
