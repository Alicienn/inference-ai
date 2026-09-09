# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
assert "needs no training: it can be frozen" in lines[568], lines[568]
assert "is roughly scale-invariant" in lines[577], lines[577]
lines[568:578] = [
 "needs no training. Freezing it on the first $256$ tokens costs $+0.0032$\\,nat",
 "against \\emph{maxip} on the same chunk---better than recomputing the basis on",
 "the full chunk ($+0.0551$); freezing on $128$ tokens costs $+0.0476$. A basis",
 "estimated on a different domain transfers: evaluated across academic prose,",
 "encyclopedia and source code, cross-domain bases cost $0.007$--$0.075$\\,nat",
 "against $1.52$--$2.05$ for random selection. As the candidate pool grows from",
 "$16$ to $32$ to $64$ blocks ($T=512,1024,2048$), the index costs $+0.055$,",
 "$+0.078$ and $+0.132$\\,nat, that is $3.0\\%$, $4.2\\%$ and $7.5\\%$ of the gap",
 "between \\emph{maxip} and random selection: the index keeps $93$--$97\\%$ of the",
 "selectable gain, while the cost of the sparse budget itself stays near",
 "$0.16$\\,nat as the context quadruples. Distilling the basis to imitate",
 "\\emph{maxip} does \\emph{not} help: after correcting for overfitting it remains",
 "$0.014$\\,nat behind the training-free PCA basis; this last comparison is the one",
 "still being re-measured.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
