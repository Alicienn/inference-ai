# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
lines = p.read_text(encoding="utf-8").split("\n")
def rep(old, new):
    for i, L in enumerate(lines):
        if old in L:
            lines[i] = L.replace(old, new); return True
    print("  MANQUE :", old[:70]); return False
ok = True
# 1) tableau : trois lignes PCA
ok &= rep("PCA $d'=16$           & $25\\%$    & $2.8336$ & $+0.0084$ \\\\",
          "PCA $d'=16$, causal   & $25\\%$    & $2.8302$ & $+0.0049$ \\\\")
ok &= rep("PCA $d'=8$            & $12.5\\%$  & $2.8575$ & $+0.0323$ \\\\",
          "PCA $d'=8$, causal    & $12.5\\%$  & $2.8359$ & $+0.0107$ \\\\\n"
          "PCA $d'=8$, transductive & $12.5\\%$ & $2.8575$ & $+0.0323$ \\\\")
# 2) resume
ok &= rep("within $0.032$\\,nat ($0.0175$ on Qwen2.5-0.5B), needs no training, survives freezing",
          "within $0.011$\\,nat ($0.022$ on Qwen2.5-0.5B) when the basis is fit on the first\n$256$ keys, needs no training, survives freezing")
# 3) phrase du corps
ok &= rep("costs $0.032$\\,nat against", "costs $0.0107$\\,nat against")
ok &= rep("bytes) the cost falls to $0.008$. The same comparison on Qwen2.5-0.5B---a",
          "bytes) the cost falls to $0.0049$ with a causal basis. The same comparison on Qwen2.5-0.5B---a")
# 4) nouveau paragraphe apres le cout de la sparsite
done = False
for i, L in enumerate(lines):
    if "deployment must trade against the byte savings." in L:
        lines[i+1:i+1] = [
            "",
            "\\paragraph{The basis must not see the future.} The harness fit the PCA",
            "basis on all $T$ keys of the chunk, including keys that lie \\emph{after} the",
            "query. We audited this by fitting the basis on the first $256$ keys only---a",
            "causal, streaming-compatible choice---and evaluating on the same chunks. The",
            "transductive basis is \\emph{worse}, by $0.0216$\\,nat on SmolLM2-135M and",
            "$0.0225$ on Qwen2.5-0.5B: letting the index see the future does not help it,",
            "it hurts it, presumably because the chunk-wide key distribution is dominated",
            "by distant tokens. The numbers in the table are therefore the causal ones,",
            "with the transductive row kept as a control; this is also why the frozen",
            "basis of the previous paragraph beat the full-chunk basis.",
        ]
        done = True; break
if not done:
    print("  MANQUE : paragraphe audit"); ok = False
p.write_text("\n".join(lines), encoding="utf-8")
print("ok =", ok, "|", len("\n".join(lines)), "octets")
