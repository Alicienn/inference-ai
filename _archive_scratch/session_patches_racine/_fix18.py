# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
def rep(old, new):
    for i, L in enumerate(lines):
        if old in L:
            lines[i] = L.replace(old, new); return True
    print("  MANQUE :", old[:60]); return False
ok = True
ok &= rep("matches an exact-mass oracle to within $0.0002$\\,nat. Projecting keys onto",
          "matches an exact-mass oracle to within $0.0002$\\,nat on SmolLM2-135M and\n$0.001$\\,nat on Qwen2.5-0.5B. Projecting keys onto")
ok &= rep("within $0.032$\\,nat, needs no training, survives freezing",
          "within $0.032$\\,nat ($0.0175$ on Qwen2.5-0.5B), needs no training, survives freezing")
ok &= rep("bytes) the cost falls to $0.008$. A random projection at the same",
          "bytes) the cost falls to $0.008$. The same comparison on Qwen2.5-0.5B---a\n"
          "different architecture, $14$ query heads and $2$ KV heads---gives $+0.0175$\\,nat\n"
          "at $d'=8$ and $-0.0037$\\,nat at $d'=16$, the latter within noise of\n"
          "\\emph{maxip}, whose own gap to the oracle there is $0.001$\\,nat. A random\n"
          "projection at the same")
p.write_text("\n".join(lines), encoding="utf-8")
print("ok =", ok, "|", len("\n".join(lines)), "octets")
