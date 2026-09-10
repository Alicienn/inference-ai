# -*- coding: utf-8 -*-
import pathlib, shutil
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\17_papier\asp.tex")
bak = p.with_name("_asp_v1_backup.tex")
if not bak.exists():
    shutil.copy(p, bak)
t = p.read_text(encoding="utf-8")

# --- 1) abstract ---
s = t.index("\\begin{abstract}"); e = t.index("\\end{abstract}") + len("\\end{abstract}")
NEWABS = r"""\begin{abstract}
Block-sparse attention makes long-context decoding tractable by reading only a
fraction of the KV cache, but the \emph{selector}---the mechanism that decides
which blocks to read---has itself become the dominant cost. We study
\emph{Adaptive Sequential Precision} (ASP), a two-pass selector that scores all
blocks with a coarse per-block summary, then rescores only the survivors with a
finer one, and we formalise it as a \emph{precision-allocation bandit}: a
best-arm-identification problem in which the measurement budget is spent in
\emph{bits per arm} along a rate--distortion curve rather than in repeated
stochastic pulls.

Our first headline result is negative and, we argue, instructive. On Qwen3-8B, ASP
reduces selector KV traffic by $13\text{--}25\times$ yet is $1.2\text{--}1.7\times$
\emph{slower} end-to-end. We explain the gap in three measured steps. First, the
irregular gather that ASP requires is essentially free on datacenter hardware:
across three memory architectures (LPDDR5, GDDR6, HBM2e) scattered reads reach
$98\text{--}104\%$ of contiguous bandwidth once each gathered chunk exceeds
$\approx$1\,KiB, and on A100 the granularity threshold disappears entirely.
Second, fusing the two passes into a single kernel with interleaved stratified
selection yields $1.57\times$ over the two-kernel form on an H200 (15/15
configurations), but only when the per-stratum quota is small ($q\!\approx\!8$);
at $q=64$ fusion loses by $3\times$. Third, the residual end-to-end gap is
per-layer kernel-launch overhead: an unfused PyTorch selector costs a flat
$\approx$20\,ms per decode step, which alone exceeds the entire dense attention
time ($2\text{--}11$\,ms at $16$k--$262$k tokens, itself running at $84\%$ of
peak bandwidth).

A second, later result revises what we believe about selector \emph{quality}, and
we report it with the same prominence because it changes how the first result
should be read. An audit found that our perplexity harness omitted the causal
mask, so every quality measurement in the first version was contaminated. Under a
corrected harness, the mean-pooled block key that ASP scores with ranks blocks
\emph{worse than chance}, whereas the maximum of the per-key inner products inside
a block matches an exact-mass oracle to within $0.003$\,nat. Projecting keys onto
eight principal components per head---$12.5\%$ of key bytes---preserves this to
within $0.023$--$0.033$\,nat on two models, needs no training, survives freezing
at prefill, and transfers across texts and domains. The near-tie that we
previously took to cap selector quality is an artefact of the estimator, not of
the problem.

We additionally report two \emph{refuted} predictive models of the gather
granularity threshold---one based on Little's law, one on DRAM page
invariance---and a measurement-driven law relating selection recall to
estimator noise, whose residuals are explained by Jensen's inequality over
per-query noise dispersion ($r=0.96$). We argue that convergent empirical
behaviour across three memory architectures without a reliable predictive model is itself a
result worth documenting.
\end{abstract}"""
t = t[:s] + NEWABS + t[e:]

# --- 2) nouvelle sous-section avant la Discussion ---
NEWSUB = r"""%======================================================================
\subsection{A corrected selector: the maximum, not the mean}
\label{sec:corrected}

An audit of our quality harness, performed after the first version of this paper,
changed our reading of the selector. We report the correction with the same
prominence as the negative result, because it moves the paper's centre of gravity
from \emph{how much} the selector costs to \emph{what} it computes.

\paragraph{The harness was wrong.} Every perplexity measurement in the first
version used sparse masks that omitted the causal bound $j\le p$. The model could
therefore read \emph{future} tokens through the local-window branch. This inverted
the ranking of selectors: the window-only baseline appeared to beat dense
attention, adding blocks appeared to hurt monotonically, and an oracle scoring
blocks by exact attention mass appeared to be the \emph{worst} selector, because
each additional block diluted the leak. The corrected harness carries a
per-position invariance test---the loss on a prefix must equal the loss on the
prefix alone---whose measured discrepancy is $2.9\times10^{-5}$.

\paragraph{What the corrected measurements say.} With the mask fixed, the problem
is far better behaved than the first version suggested (Table~\ref{tab:corrected}).
Scoring a block by the maximum inner product over its keys (\emph{maxip}) matches
an exact-mass oracle to within $0.003$\,nat, on two models and for
$m\in\{1,4,8\}$ selected blocks. The mean-pooled block key that ASP uses ranks
blocks worse than chance on both models. This explains, after the fact, why four
independent refinements of the summary---optimal-weight quadrature, certified
selection, calibrated coresets, adaptive $k$---all failed to improve ranking:
they were refining a statistic that had already destroyed the signal.

\begin{table}[t]
\centering\small
\caption{Corrected selector quality. SmolLM2-135M, $T=512$, window $128$, block
$32$, $m=2$, means over three corpus chunks; only intra-chunk comparisons are
interpretable, since the spread across chunks ($\approx1$\,nat) dwarfs the
differences between selectors. ``Bytes'' is the index per key relative to the full
key vector, at equal per-component precision.}
\label{tab:corrected}
\begin{tabular}{lrrr}
\toprule
selector (score per block) & bytes & loss & $\Delta$ vs \emph{maxip} \\
\midrule
dense attention       & $100\%$   & $2.6108$ & $-0.034$ \\
\emph{maxip}          & $100\%$   & $2.6453$ & --- \\
PCA $d'=16$           & $25\%$    & $2.6572$ & $+0.012$ \\
PCA $d'=8$            & $12.5\%$  & $2.6787$ & $+0.033$ \\
random blocks         & ---       & $2.8211$ & $+0.176$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{A compact index, with no training.} The selector must score every block,
so its own bytes matter. Projecting queries and keys onto $d'=8$ principal
components per head---$12.5\%$ of key bytes---costs $0.033$\,nat against
\emph{maxip} on SmolLM2-135M and $0.023$ on Qwen2.5-0.5B; at $d'=16$ ($25\%$ of
bytes) the cost falls to $0.012$ and $0.003$. A random projection at the same
budget fails ($3.52$ versus $3.12$ for chance), so the basis matters. The basis
needs no training: it can be frozen on the first $256$ tokens ($2.9796$, within
$0.018$\,nat of \emph{maxip}), it transfers from a different text ($3.0326$), and
the spread between bases estimated on different domains (academic prose,
encyclopedia, source code) is $0.01$--$0.04$\,nat against $0.10$--$0.22$ to
chance. Distilling the basis to imitate \emph{maxip} does \emph{not} help: after
correcting for overfitting it remains $0.014$\,nat behind the training-free PCA
basis. As the number of candidate blocks grows from $12$ to $28$ to $60$, the
index retains $81\%$, $81\%$ and $78\%$ of the selectable gain, so the advantage
is roughly scale-invariant but the absolute cost in nats grows.

\paragraph{What survives from the first version.} The latency findings are kernel-
and memory-level measurements and are unaffected by the harness bug: the
irregular gather remains free on datacenter hardware, fusion still wins at small
quota, and an unfused selector still costs about $20$\,ms per decode step. What
does \emph{not} survive is the inference that the approach is near its quality
ceiling. The ceiling was the estimator, and it is fixable for $12.5\%$ of the key
bytes.

"""
anchor = "%======================================================================\n\\section{Discussion}\\label{sec:disc}"
assert anchor in t
t = t.replace(anchor, NEWSUB + anchor, 1)

# --- 3) paragraphe "Where the effort should go" ---
old_disc_start = "\\paragraph{Where the effort should go.}"
old_disc_end = "\\paragraph{Sparse attention and speculation conflict.}"
i0 = t.index(old_disc_start); i1 = t.index(old_disc_end)
NEWDISC = r"""\paragraph{Where the effort should go.} The corrected measurements
(Section~\ref{sec:corrected}) reverse one of our earlier priorities. We previously
concluded that making $k$ larger and cheaper was worth more than making the
selector smarter, because the near-tie structure appeared to cap selector quality.
That ceiling was an artefact: with the maximum score per block the selector sits
within $0.003$\,nat of an exact-mass oracle, and a training-free PCA index reaches
it within $0.033$\,nat at $12.5\%$ of key bytes. The remaining question is
therefore mechanical rather than statistical---whether reading $12.5\%$ of the key
bytes in an index becomes time---which the fused kernel of
Section~\ref{sec:fusion} suggests it can, but which no end-to-end measurement in
this paper establishes.

"""
t = t[:i0] + NEWDISC + t[i1:]

# --- 4) conclusion ---
old_con_start = "Two-pass precision allocation reduces"
old_con_end = "cost nothing on datacenter hardware."
i0 = t.index(old_con_start); i1 = t.index(old_con_end) + len(old_con_end)
NEWCON = r"""Two-pass precision allocation reduces selector traffic by $13$--$25\times$ on a
real model, and that reduction does not become a speedup unless two conditions
hold: the selector must be a single fused kernel, and the surrounding engine must
be memory-bound. We measured the first condition being met ($1.57\times$ over the
two-kernel form on H200) and the second being violated ($3.2\times$ from the
memory floor). The irregular gather, which motivated this study, turns out to cost
nothing on datacenter hardware. A corrected quality harness then changed what we
believe the selector is doing: the mean-pooled key that ASP scores with ranks
blocks worse than chance, whereas a maximum over keys projected onto eight
principal components per head recovers the exact-mass oracle to within
$0.033$\,nat at $12.5\%$ of key bytes, with no training."""
t = t[:i0] + NEWCON + t[i1:]

p.write_text(t, encoding="utf-8")
print("asp.tex revise :", len(t), "octets (backup _asp_v1_backup.tex)")
