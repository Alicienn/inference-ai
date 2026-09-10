# -*- coding: utf-8 -*-
"""METRIQUE UNIFIANTE v2 : masse retenue S et distance L1 sparse/dense.
6 configs sur le banc GPT-2, frac=0,375. On teste la correlation avec Delta_loss."""
import math, pathlib, torch, torch.nn.functional as F, random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "gpt2"
T, NBASE, LB, NT, SPAN, DMAX = 256, 128, 4, 4, 64, 64
QPOS = list(range(193, 256))
CFG = {"mode": "dense", "W": 32, "m": 16, "dp": 8, "kagg": "max", "bagg": "max"}
BASE = {}
ACC = {"peak": 0, "peakn": 0, "S": 0.0, "L1": 0.0, "nl": 0}

tok = GPT2Tokenizer.from_pretrained(MID)
model = GPT2LMHeadModel.from_pretrained(MID, attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]


def make_fwd(li):
    def fwd(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
            encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False,
            output_attentions=False, **kw):
        bsz, qn, _ = hidden_states.shape
        x = self.c_attn(hidden_states)
        q, k, v = x.split(self.split_size, dim=2)
        hd, nh = self.head_dim, self.num_heads
        q = q.view(bsz, qn, nh, hd).permute(0, 2, 1, 3)
        k = k.view(bsz, qn, nh, hd).permute(0, 2, 1, 3)
        v = v.view(bsz, qn, nh, hd).permute(0, 2, 1, 3)
        h = q.shape[1]
        if CFG["mode"] == "collect":
            K = k[0].float(); Kc = K - K.mean(2, keepdim=True)
            U, _, _ = torch.linalg.svd(Kc.transpose(1, 2), full_matrices=False)
            BASE[li] = U[:, :, :DMAX].contiguous()
        S = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        j = torch.arange(qn)
        S = S.masked_fill(j[None, None, :] > j[None, :, None], float("-inf"))
        A = torch.softmax(S, dim=-1)
        del S
        W, m = CFG["W"], CFG["m"]
        if CFG["mode"] == "pca" and m > 0 and qn == T:
            A3 = A[0]
            nb = qn // LB
            U = BASE[li][:, :, :CFG["dp"]]
            qp = torch.einsum("hqd,hde->hqe", q[0], U)
            kp = torch.einsum("hkd,hde->hke", k[0], U)
            prod = qp[:, :, None, :] * kp[:, None, :, :]
            sc = prod.amax(-1) if CFG["kagg"] == "max" else prod.sum(-1)
            del prod, qp, kp
            v4 = sc[:, :, :nb * LB].view(h, qn, nb, LB)
            if CFG["bagg"] == "max":
                B = v4.amax(-1)
            elif CFG["bagg"] == "sum":
                B = v4.sum(-1)
            else:  # LSE : interpole entre max (beta grand) et somme (beta petit)
                lam = float(CFG["bagg"][3:])
                beta = lam / v4.std().clamp_min(1e-6)
                B = torch.logsumexp(beta * v4, dim=-1) / beta
            del v4
            bid = torch.arange(nb)
            elig = (bid[None, :] * LB + LB - 1) <= (torch.arange(qn) - W)[:, None]
            _, sel = torch.topk(B * elig[None], min(m, nb), dim=2)
            keys = (sel[..., None] * LB + torch.arange(LB)).reshape(h, qn, -1)
            mask = torch.zeros(h, qn, qn, dtype=torch.bool)
            mask.scatter_(2, keys, True)
            mask |= (j[None, None, :] > (j[None, :, None] - W))
            pk = A3.argmax(-1)
            pkb = pk // LB
            hit = (sel == pkb[..., None]).any(-1)
            inwin = pk > (torch.arange(qn)[None, :] - W)
            keep = hit | inwin
            ACC["peak"] += int(keep[:, QPOS].sum())
            ACC["peakn"] += keep[:, QPOS].numel()
            A3s = A3 * mask
            Sm = A3s.sum(-1)                                   # masse retenue
            A3n = A3s / Sm.clamp(min=1e-9).unsqueeze(-1)
            L1 = (A3n - A3).abs().sum(-1)                       # distance L1
            ACC["S"] += float(Sm[:, QPOS].mean())
            ACC["L1"] += float(L1[:, QPOS].mean())
            ACC["nl"] += 1
            A = A3n.unsqueeze(0)
            del mask, B, A3s, A3n
        o = torch.matmul(A, v)
        del A
        o = o.permute(0, 2, 1, 3).reshape(bsz, qn, nh * hd)
        return self.c_proj(o), None
    return fwd


for li, blk in enumerate(model.transformer.h):
    blk.attn.forward = make_fwd(li).__get__(blk.attn, type(blk.attn))

CFG["mode"] = "collect"
with torch.no_grad():
    model(ALL[:NBASE][None])
print("base collectee", flush=True)

seqs = []
rng = random.Random(11)
for i in range(NT):
    s = rng.randint(2000, len(ALL) - SPAN - 8)
    seqs.append(ALL[s:s + SPAN].clone().repeat(4))

log = OUT / "gpt2_lse.txt"
log.write_text(f"{MID} | banc induction | balayage LSE | S = masse retenue, L1 = |sparse-dense|\n"
               "m     bagg   perte      Delta     pic%   S       L1\n", encoding="utf-8")
rows = []
D = 0.0


def run(dp, kagg, bagg, m=16, W=32):
    global D
    if dp is None:
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
    else:
        CFG["mode"] = "pca"; CFG["dp"] = dp; CFG["kagg"] = kagg; CFG["bagg"] = bagg
        CFG["W"] = W; CFG["m"] = m
        for kk in ACC: ACC[kk] = 0
    l4 = 0.0
    for seq in seqs:
        with torch.no_grad():
            logits = model(seq[None]).logits[0]
        ls = F.cross_entropy(logits[:-1], seq[1:], reduction="none")
        l4 += float(ls[193:T - 1].mean())
    l4 /= NT
    if dp is None:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"---  dense  ---  {l4:9.5f}  ---       ---     ---     ---\n")
        print(f"dense: {l4:.5f}", flush=True)
        return l4
    nl = max(ACC["nl"], 1)
    pk = 100.0 * ACC["peak"] / max(ACC["peakn"], 1)
    Sm, L1 = ACC["S"] / nl, ACC["L1"] / nl
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{m:4d}  {bagg:6s}  {l4:9.5f}  {l4 - D:+.5f}  {pk:5.1f}  "
                f"{Sm:.4f}  {L1:.4f}\n")
    print(f"m={m} {bagg}: perte={l4:.5f} Delta={l4-D:+.5f} pic={pk:.1f}% "
          f"S={Sm:.4f} L1={L1:.4f}", flush=True)
    rows.append((f"m{m}-{bagg}", l4 - D, pk, Sm, L1))
    return l4


D = run(None, None, None)
for mm in (8, 16, 24):
    print(f"### frac lue = {(32 + mm*4)/256:.3f} (m={mm}, W=32)", flush=True)
    for bg in ("max", "lse2", "lse1", "lse0.5", "lse0.25", "sum"):
        run(8, "max", bg, m=mm, W=32)

# correlations
def pear(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def spear(xs, ys):
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(idx):
            r[i] = pos
        return r
    return pear(rank(xs), rank(ys))


dl = [r[1] for r in rows]
print("\n--- correlations avec Delta_loss (6 configs) ---", flush=True)
for name, vals in [("1 - pic%", [100 - r[2] for r in rows]),
                   ("S", [r[3] for r in rows]),
                   ("L1", [r[4] for r in rows]),
                   ("-S", [-r[3] for r in rows])]:
    print(f"{name:10s} Pearson={pear(vals, dl):+.3f}  Spearman={spear(vals, dl):+.3f}",
          flush=True)
with log.open("a", encoding="utf-8") as f:
    f.write("\ncorrelations avec Delta_loss :\n")
    for name, vals in [("1-pic%", [100 - r[2] for r in rows]), ("S", [r[3] for r in rows]),
                       ("L1", [r[4] for r in rows])]:
        f.write(f"{name:8s} Pearson={pear(vals, dl):+.3f} Spearman={spear(vals, dl):+.3f}\n")
print("FIN")
