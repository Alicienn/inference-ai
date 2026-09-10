# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
hit = None
for i, L in enumerate(lines):
    if "way, so it is best read as free rather than as a fixed penalty." in L:
        hit = i; break
assert hit is not None
lines[hit+1:hit+1] = [
 "",
 "Pushing the grid further shows where the effect saturates. On Qwen, $L_b=2$ and",
 "$L_b=1$ give $-0.003$ and $-0.004$\\,nat, no better than $L_b=4$ ($-0.003$). At",
 "$L_b=1$ the blocks degenerate to single keys, so the selector becomes exact",
 "top-$m$ selection and the measurement is the true per-key oracle at that budget:",
 "the block structure costs nothing once the grid is fine enough. The practical rule",
 "is $L_b\\approx4$--$8$, and below it the quota $m$ grows without buying anything.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
