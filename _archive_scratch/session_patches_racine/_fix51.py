# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.endswith("drives the negative headline is untouched by it."):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
lines[i0+1:i0+1] = [
 "",
 "\\paragraph{Where the gain comes from, and what this testbed cannot test.} Sweeping the",
 "read fraction with the deployable index at $T=1024$ (window and quota scaled together,",
 "$L_b=4$) the loss gain over dense grows monotonically as less is read: $-0.036$ at",
 "$37.5\\%$, $-0.077$ at $24.7\\%$, $-0.090$ at $12.3\\%$ and $-0.144$ at $6.0\\%$. Read",
 "naively this says sparse is better the sparser it is. Two control probes show why that",
 "reading is unsafe. A synthetic needle of $16$ tokens placed twice gives a dense",
 "attention mass of $0.0395$ on the needle keys, against $20/512=0.039$ for a uniform",
 "distribution; a natural span repeated four times, where induction is demonstrably",
 "active (the loss on the fourth copy is $0.395$\\,nat below the first), gives a dense",
 "mass of $0.263$ on the source copy against $0.250$ uniform. In both cases distant",
 "content carries no attention above chance, so there is no concentrated component for a",
 "selector to lose---consistent with the sink-plus-background structure. The local",
 "testbed therefore cannot decide whether sparsity preserves retrieval; that needs a",
 "model or a corpus in which distant attention concentrates.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok :", len("\n".join(lines)), "octets")
