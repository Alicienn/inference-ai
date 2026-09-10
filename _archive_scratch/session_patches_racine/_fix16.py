# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
t = p.read_text(encoding="utf-8")
lines = t.split("\n")

def rep(old, new, once=True):
    n = 0
    for i, L in enumerate(lines):
        if old in L:
            lines[i] = L.replace(old, new); n += 1
            if once: break
    if n == 0:
        print("  MANQUE :", old[:70]); return False
    return True

ok = True
ok &= rep("to within $0.003$\\,nat. Projecting keys onto", "to within $0.0002$\\,nat. Projecting keys onto")
ok &= rep("to within $0.003$\\,nat, on two models and for", "to within $0.0002$\\,nat (three corpus chunks).")
ok &= rep("the cost falls to $0.012$ and $0.003$. A random projection at the same", "the cost falls to $0.008$.")
ok &= rep("within $0.003$\\,nat of an exact-mass oracle", "within $0.0002$\\,nat of an exact-mass oracle")
ok &= rep("within $0.023$--$0.033$\\,nat on two models, needs no training", "within $0.032$\\,nat, needs no training")
ok &= rep("costs $0.033$\\,nat against", "costs $0.032$\\,nat against")
ok &= rep("\\emph{maxip} on SmolLM2-135M and $0.023$ on Qwen2.5-0.5B; at $d'=16$ ($25\\%$ of", "\\emph{maxip}; at $d'=16$ ($25\\%$ of")
ok &= rep("it within $0.033$\\,nat at $12.5\\%$ of key bytes. The remaining question is", "it within $0.032$\\,nat at $12.5\\%$ of key bytes. The remaining question is")
ok &= rep("$0.033$\\,nat at $12.5\\%$ of key bytes, with no training.", "$0.032$\\,nat at $12.5\\%$ of key bytes, with no training.")
ok &= rep("dense attention       & $100\\%$   & $2.6108$ & $-0.034$ \\\\", "dense attention       & $100\\%$   & $2.6108$ & $-0.2145$ \\\\")
ok &= rep("\\emph{maxip}          & $100\\%$   & $2.6453$ & --- \\\\",
          "\\emph{maxip}          & $100\\%$   & $2.8253$ & --- \\\\\n"
          "exact-mass oracle     & $100\\%$   & $2.8255$ & $+0.0002$ \\\\")
ok &= rep("PCA $d'=16$           & $25\\%$    & $2.6572$ & $+0.012$ \\\\", "PCA $d'=16$           & $25\\%$    & $2.8336$ & $+0.0084$ \\\\")
ok &= rep("PCA $d'=8$            & $12.5\\%$  & $2.6787$ & $+0.033$ \\\\", "PCA $d'=8$            & $12.5\\%$  & $2.8575$ & $+0.0323$ \\\\")
ok &= rep("random blocks         & ---       & $2.8211$ & $+0.176$ \\\\", "random blocks         & ---       & $4.5155$ & $+1.6903$ \\\\")

# 2e bug : corriger le nombre + inserer le paragraphe
for i, L in enumerate(lines):
    if "prefix alone---whose measured discrepancy is" in L:
        lines[i] = L.replace("$2.9\\times10^{-5}$", "$1.6\\times10^{-5}$")
        newp = ["", "\\paragraph{A second bug, found by independent reimplementation.}",
                "Reimplementing the sparse path as explicit per-query loops---recomputing",
                "the projections, the block scores and the mask from the weights---revealed",
                "a second defect. The sentinel value $-1$, used when fewer than $m$ blocks",
                "are eligible, was clamped to index $0$ before being scattered into the",
                "selection mask, so block~$0$ was marked selected whenever fewer than $m$",
                "blocks were eligible---that is, for the first $\\approx$160 positions of",
                "every chunk. Block~$0$ contains the attention sink, so the defect",
                "\\emph{flattered} every sparse selector. Correcting it raises the cost of",
                "sparsity from $0.034$ to $0.215$\\,nat and makes the random control",
                "$10\\times$ more discriminating ($+1.69$\\,nat), while the estimator and",
                "index comparisons below barely move---which is why we report both sets of",
                "numbers. The two implementations agree to $1.2\\times10^{-8}$ on the",
                "attention output at the query where they were compared, and a structural",
                "check confirms the eligible-position count ($101$ at $p=100$, $192$ at",
                "$p=300$)."]
        lines[i+1:i+1] = newp
        break
else:
    print("  MANQUE : paragraphe 2e bug"); ok = False

# cout de la sparsite, apres le paragraphe 'ceiling'
for i, L in enumerate(lines):
    if "ceiling was the estimator" in L:
        j = i
        while j < len(lines) and lines[j].strip() != "":
            j += 1
        lines[j:j] = ["", "What the corrected harness does show, and the earlier one hid, is that the",
                      "sparse budget itself---window $128$ plus two blocks of $32$, $37.5\\%$ of",
                      "the keys---costs $0.21$\\,nat against dense attention. That cost belongs",
                      "to the sparsity budget, not to the selector, and it is the quantity a",
                      "deployment must trade against the byte savings."]
        break
else:
    print("  MANQUE : cout de la sparsite"); ok = False

p.write_text("\n".join(lines), encoding="utf-8")
print("ecrit :", len("\n".join(lines)), "octets | ok =", ok)
