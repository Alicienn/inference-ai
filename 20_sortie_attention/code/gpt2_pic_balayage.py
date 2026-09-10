# -*- coding: utf-8 -*-
"""L'index peut-il atteindre 99,5 % de retention du PIC ?
Balayage (d', fraction lue) sur le banc GPT-2 valide. Base SVD complete collectee
une fois, tranchee a d'. Lb=4, W=T*frac/3, m*Lb=2T*frac/3."""
import math, pathlib, torch, torch.nn.functional as F, random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "gpt2"
T, NBASE, LB, NT, SPAN, DMAX = 256, 128, 4, 4, 64, 64
QPOS = list(range(193, 256))
CFG = {"mode": "dense", "W": 32, "m": 16, "dp": 8}
BASE = {}
ACC = {"peak": 0, "peakn": 0}

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
            sc = (qp[:, :, None, :] * kp[:, None, :, :]).amax(-1)
            B = sc[:, :, :nb * LB].view(h, qn, nb, LB).amax(-1)
            del qp, kp, sc
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
            A = (A3 * mask).unsqueeze(0)
            A = A / A.sum(-1, keepdim=True).clamp(min=1e-9)
            del mask, B
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
print("base SVD complete collectee", flush=True)

seqs = []
rng = random.Random(11)
for i in range(NT):
    s = rng.randint(2000, len(ALL) - SPAN - 8)
    seqs.append(ALL[s:s + SPAN].clone().repeat(4))

log = OUT / "gpt2_pic_balayage.txt"
log.write_text(f"{MID} | banc induction | Lb={LB} | base PCA sur {NBASE} tok | "
               f"cible : retention du pic >= 99,5 %\n"
               "dp   frac   W    m    perte_copie4  Delta   pic_retenu%\n", encoding="utf-8")
D = 0.0


def run(dp, frac):
    global D
    if frac is None:
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
    else:
        CFG["mode"] = "pca"; CFG["dp"] = dp
        CFG["W"] = max(8, int(T * frac / 3)); CFG["m"] = max(1, int((T * frac * 2 / 3) // LB))
        for kk in ACC: ACC[kk] = 0
    l4 = 0.0
    for seq in seqs:
        with torch.no_grad():
            logits = model(seq[None]).logits[0]
        ls = F.cross_entropy(logits[:-1], seq[1:], reduction="none")
        l4 += float(ls[193:T - 1].mean())
    l4 /= NT
    if frac is None:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"---  dense  ---  ---  {l4:12.5f}  ---     ---\n")
        print(f"dense: {l4:.5f}", flush=True)
        return l4
    pk = 100.0 * ACC["peak"] / max(ACC["peakn"], 1)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{dp:3d}  {frac:.3f}  {CFG['W']:4d} {CFG['m']:4d}  {l4:12.5f}  "
                f"{l4 - D:+.5f}  {pk:6.1f}\n")
    print(f"dp={dp} frac={frac:.3f} W={CFG['W']} m={CFG['m']} perte={l4:.5f} "
          f"Delta={l4-D:+.5f} pic={pk:.1f}%", flush=True)
    return l4


D = run(8, None)
for fr in (0.25, 0.375, 0.5, 0.75):
    run(8, fr)
for dp in (16, 32):
    run(dp, 0.375)
print("FIN")
