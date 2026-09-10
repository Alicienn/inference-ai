# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if "on it. On this bench we can finally ask" in L:
        i0 = i; break
assert i0 is not None, "debut introuvable"
i1 = None
for i in range(i0, len(lines)):
    if "opposite ends of the budget axis." in lines[i]:
        i1 = i; break
assert i1 is not None, "fin introuvable"
new = [
 "on it. At a $37.5\\%$ read budget the oracle, which selects blocks by true attention",
 "mass, costs $0.007$\\,nat; the deployable index costs $0.226$. The obvious",
 "explanation---that the index fails to retain the attention peak---does not survive.",
 "Across six index variants at the same budget, the peak-retention rate has rank",
 "correlation $0.14$ with the loss, while the \\emph{retained attention mass} $S$ has",
 "rank correlation $-1.00$: $S=0.857$ costs $0.226$\\,nat, $S=0.789$ costs $1.17$ and",
 "$S=0.553$ costs $4.64$. The objective is the one the flowing-text regime already",
 "used---captured mass---and so is the aggregation that maximises it: a maximum over",
 "projections and over keys. Replacing the maximum over projections by a sum drops $S$",
 "to $0.789$ and inflates the loss fivefold; replacing the maximum over keys by a sum",
 "drops $S$ to $0.553$ and inflates it twentyfold. Budget buys $S$: the loss on the",
 "retrieval task falls through $0.643$, $0.288$, $0.132$ and $-0.001$\\,nat at $25\\%$,",
 "$37.5\\%$, $50\\%$ and $75\\%$ of keys, and at $75\\%$ it matches dense. Projection rank",
 "buys little: at $37.5\\%$, $d'=32$ with max scoring reaches $S=0.867$ against $0.856$",
 "for $d'=8$. On retrieval tasks ASP therefore saves about $1.3\\times$, not the",
 "$16\\times$ it saves on flowing text---the two regimes sit at opposite ends of the",
 "budget axis, but they share one objective, captured attention mass.",
]
lines[i0:i1+1] = new
p.write_text("\n".join(lines), encoding="utf-8")
print("ok :", len("\n".join(lines)), "octets")
