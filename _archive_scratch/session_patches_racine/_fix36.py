# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
hit = None
for i, L in enumerate(lines):
    if "index. Locality alone buys almost nothing; the two-pass selection does the work." in L:
        hit = i; break
assert hit is not None
lines[hit+1:hit+1] = [
 "",
 "Pushing the budget down sharpens the contrast. At $128$ keys---a quarter of the",
 "context---the two-pass selector costs $0.033$\\,nat, while a pure window of exactly",
 "the same size costs $1.68$\\,nat: a factor of $50$ at identical bytes. The index's",
 "own cost rises from $0.008$ at $192$ keys to $0.022$ at $128$, because with fewer",
 "blocks admitted each selection error weighs more. The frontier is shallow: going",
 "from $192$ to $128$ keys costs $0.012$\\,nat, so a quarter of the context is enough",
 "for the selector to stay within a few hundredths of dense attention.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
