# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.endswith("model or a corpus in which distant attention concentrates."):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
lines[i0+1:i0+1] = [
 "The same probe on Qwen2.5-0.5B does not repair this: dense induction is absent",
 "there (the loss on the fourth copy is $0.001$\\,nat \\emph{above} the first) and the",
 "attention on the source copy is $0.204$, \\emph{below} the $0.250$ uniform level, while",
 "the nearest copy receives $0.287$---a mild recency tilt, not retrieval. Scaling the",
 "probe model from $135$\\,M to $0.5$\\,B does not produce a concentrated distant",
 "component.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok :", len("\n".join(lines)), "octets")
