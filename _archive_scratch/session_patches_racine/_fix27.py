# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
assert "$p=300$)." in lines[526], lines[526]
lines[527:527] = [
 "",
 "\\paragraph{Extending the verification.} We then re-ran the same independent route",
 "on the second model and on the paths the first pass had not touched: Qwen2.5-0.5B",
 "($14$ query heads, $2$ KV heads, hence grouped-query attention), the causal PCA",
 "selector, and rotary embeddings. The reference recomputes the rotations by hand",
 "from the half-split convention, obtains the basis by a singular value",
 "decomposition of the fit keys rather than an eigendecomposition of their",
 "covariance, ranks blocks by an explicit sort rather than \\texttt{topk}, and maps",
 "query head $h$ to KV head $h//7$. Over $84$ comparisons ($2$ layers, $3$ query",
 "positions, $14$ heads), the selected block sets and the masks are \\emph{identical}",
 "in every case, and the attention outputs agree to $1.4\\times10^{-5}$ relative.",
 "This time the audit confirmed the harness rather than breaking it---the outcome a",
 "verification pass must be able to produce, and the reason we kept running it.",
]
p.write_text("\n".join(lines), encoding="utf-8")
print("ok |", len("\n".join(lines)), "octets")
