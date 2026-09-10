# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.endswith("so the gradient shifts the mechanism without overturning it."):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
lines[i0+1:i0+1] = [
 "",
 "\\paragraph{The constant is not constant in context length.} Repeating the contrast at",
 "a fixed read fraction ($37.5\\%$), with the trapped mass measured in the same pass",
 "rather than borrowed from $\\rho$, gives at $T=1024$: $\\Delta_4=+0.022$,",
 "$\\Delta_{16}=+0.036$, $M_4=2.5\\times10^{-3}$, $M_{16}=1.2\\times10^{-2}$, hence",
 "$\\kappa=1.4$; and at $T=2048$: $\\Delta_4=-0.055$, $\\Delta_{16}=-0.018$,",
 "$M_4=1.6\\times10^{-3}$, $M_{16}=8.0\\times10^{-3}$, hence $\\kappa=5.7$. The estimate",
 "swings by an order of magnitude across context lengths and, more importantly, the",
 "sign of $\\Delta$ flips: at $T=2048$ the sparse model \\emph{beats} dense by",
 "$0.055$\\,nat at the same read fraction. So $\\Delta=\\kappa M+c$ is a local",
 "linearisation valid within a fixed configuration, not a law with a universal",
 "$\\kappa$---a context-dependent benefit of dropping the diffuse background competes",
 "with the trapped-mass cost. The sign flip is itself the useful fact: at long context",
 "the background is mostly noise and removing it helps. The contrast uses oracle",
 "top-$m$ selection, not the deployable index, and one model, so it bounds the",
 "mechanism rather than the system.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok :", len("\n".join(lines)), "octets")
