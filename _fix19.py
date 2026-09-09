# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
assert "bytes) the cost falls to $0.008$." in lines[563], lines[563]
assert "budget fails" in lines[564], lines[564]
lines[563:565] = [
 "bytes) the cost falls to $0.008$. The same comparison on Qwen2.5-0.5B---a",
 "different architecture, $14$ query heads and $2$ KV heads---gives $+0.0175$\\,nat",
 "at $d'=8$ and $-0.0037$\\,nat at $d'=16$, the latter within noise of \\emph{maxip},",
 "whose own gap to the oracle there is $0.001$\\,nat. A random projection at the same",
 "budget fails ($3.52$ versus $3.12$ for chance), so the basis matters. The basis",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("phrase reparee + Qwen ajoute |", len("\n".join(lines)), "octets")
