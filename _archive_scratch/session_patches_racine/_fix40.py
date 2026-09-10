# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
i0 = None
for i, L in enumerate(lines):
    if L.startswith("$\\kappa=11.7$ with $R^2=0.986$. The rule"):
        i0 = i; break
assert i0 is not None, "ancre introuvable"
i1 = i0
while not lines[i1].endswith("and the penalty shrinks."):
    i1 += 1
lines[i0:i1 + 1] = [
 "$\\kappa=11.7$ with $R^2=0.986$. Repeating the measurement on Qwen2.5-0.5B and",
 "GPT-2 tests whether the two constants transfer. The \\emph{shape} does: the measured",
 "penalty differences are proportional to the trapped-mass differences with $R^2$ of",
 "$0.999$ and $0.992$, and the density $\\rho$ is nearly invariant across the three",
 "models ($7.5$, $7.8$ and $8.3\\times10^{-4}$ per key, a spread under 10\\%)---the flat",
 "floor behind the window is a structural property of these attention distributions.",
 "The \\emph{scale} does not: $\\kappa$ is model-specific, $11.7$ (SmolLM2), $17.9$",
 "(GPT-2) and $33.4$ (Qwen). The rule $\\Delta\\approx\\kappa\\rho(L_b-1)/2$ therefore",
 "predicts a model-specific optimum $L_b\\lesssim1+2\\varepsilon/(\\kappa\\rho)$, which at",
 "$\\varepsilon=0.02$\\,nat gives $5.4$, $4.0$ and $2.5$ for the three models---bracketing",
 "the saturation point of $4$ observed on all three, and ordering them correctly. The",
 "same formula explains the scaling result: at a fixed fraction $\\rho$ falls as the",
 "context grows, so the optimal $L_b$ stays small and the penalty shrinks.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
