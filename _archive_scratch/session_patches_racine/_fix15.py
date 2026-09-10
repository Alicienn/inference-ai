import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
t = p.read_text(encoding="utf-8")
reps = [
 ("matches an exact-mass oracle to within $0.003$\\,nat",
  "matches an exact-mass oracle to within $0.0002$\\,nat"),
 ("preserves this to within $0.023$--$0.033$\\,nat on two models, needs no training",
  "preserves this to within $0.032$\\,nat, needs no training"),
 ("matches\nan exact-mass oracle to within $0.003$\\,nat, on two models and for\n$m\\in\\{1,4,8\\}$ selected blocks",
  "matches\nan exact-mass oracle to within $0.0002$\\,nat (three corpus chunks, SmolLM2-135M)"),
 ("costs $0.033$\\,nat against\n\\emph{maxip} on SmolLM2-135M and $0.023$ on Qwen2.5-0.5B; at $d'=16$ ($25\\%$ of\nbytes) the cost falls to $0.012$ and $0.003$.",
  "costs $0.032$\\,nat against\n\\emph{maxip}; at $d'=16$ ($25\\%$ of bytes) the cost falls to $0.008$."),
]
for a, b in reps:
    assert a in t, a[:60]
    t = t.replace(a, b)
old_tab = """dense attention       & $100\\%$   & $2.6108$ & $-0.034$ \\\\
\\emph{maxip}          & $100\\%$   & $2.6453$ & --- \\\\
PCA $d'=16$           & $25\\%$    & $2.6572$ & $+0.012$ \\\\
PCA $d'=8$            & $12.5\\%$  & $2.6787$ & $+0.033$ \\\\
random blocks         & ---       & $2.8211$ & $+0.176$ \\\\"""
new_tab = """dense attention       & $100\\%$   & $2.6108$ & $-0.2145$ \\\\
\\emph{maxip}          & $100\\%$   & $2.8253$ & --- \\\\
exact-mass oracle     & $100\\%$   & $2.8255$ & $+0.0002$ \\\\
PCA $d'=16$           & $25\\%$    & $2.8336$ & $+0.0084$ \\\\
PCA $d'=8$            & $12.5\\%$  & $2.8575$ & $+0.0323$ \\\\
random blocks         & ---       & $4.5155$ & $+1.6903$ \\\\"""
assert old_tab in t
t = t.replace(old_tab, new_tab)
# signaler le second bug et le cout de la sparsite
old_p = "The corrected harness carries a\nper-position invariance test---the loss on a prefix must equal the loss on the\nprefix alone---whose measured discrepancy is $2.9\\times10^{-5}$."
new_p = ("The corrected harness carries a\nper-position invariance test---the loss on a prefix must equal the loss on the\nprefix alone---whose measured discrepancy is $1.6\\times10^{-5}$.\n\n"
 "\\paragraph{A second bug, found by independent reimplementation.} Reimplementing\nthe sparse path as explicit per-query loops---recomputing the projections, the\nblock scores and the mask from the weights---revealed a second defect. The\nsentinel value $-1$, used when fewer than $m$ blocks are eligible, was clamped to\nindex $0$ before being scattered into the selection mask, so block~$0$ was marked\nselected whenever fewer than $m$ blocks were eligible---that is, for the first\n$\\approx$160 positions of every chunk. Block~$0$ contains the attention sink, so\nthe defect \\emph{flattered} every sparse selector. Correcting it raises the cost\nof sparsity from $0.034$ to $0.215$\\,nat and makes the random control $10\\times$\nmore discriminating ($+1.69$\\,nat); the estimator and index comparisons below are\nalmost unchanged, which is why we report both sets of numbers. The two paths now\nagree to $1.2\\times10^{-8}$ on the attention output at the query where they were\ncompared, and a structural check confirms the eligible-position count ($101$ at\n$p=100$, $192$ at $p=300$).")
assert old_p in t
t = t.replace(old_p, new_p)
# consequence : cout de la sparsite
old_c = "What does \\emph{not} survive is the inference that the approach is near its quality\nceiling. The ceiling was the estimator, and it is fixable for $12.5\\%$ of the key\nbytes."
new_c = ("What does \\emph{not} survive is the inference that the approach is near its quality\nceiling. The ceiling was the estimator, and it is fixable for $12.5\\%$ of the key\nbytes. What the corrected harness does show, and the earlier one hid, is that the\nsparse budget itself---window $128$ plus two blocks of $32$, $37.5\\%$ of the\nkeys---costs $0.21$\\,nat against dense attention. That cost belongs to the\nsparsity budget, not to the selector, and it is the quantity a deployment has to\ntrade against the byte savings.")
assert old_c in t
t = t.replace(old_c, new_c)
p.write_text(t, encoding="utf-8")
print("papier mis a jour :", len(t), "octets")
