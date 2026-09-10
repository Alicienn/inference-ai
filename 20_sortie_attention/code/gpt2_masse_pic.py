# -*- coding: utf-8 -*-
"""METRIQUE UNIFIANTE : masse du PIC conservee apres renormalisation creuse.
Banc GPT-2, frac=0,375. On mesure masse_pic_dense et masse_pic_creuse ; l'ecart
devrait expliquer la perte mieux que le simple taux de retention."""
import math, pathlib, torch, torch.nn.functional as F, random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "gpt2"
T, NBASE, LB, NT, SPAN, DMAX = 256, 128, 4, 4, 64, 64
QPOS = list(range(193, 256))
CFG = {"mode": "dense", "W": 32, "m": 16, "dp": 8, "kagg": "max", "bagg": "max"}
BASE = {}
ACC = {"peak": 0, "peakn": 0, "md": 0.0, "ms": 0.0, "nl": 0}

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
            B = v4.amax(-1) if CFG["bagg"] == "max" else v4.sum(-1)
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
            pki = pk.unsqueeze(-1)
            md = A3.gather(2, pki).squeeze(-1)                 # masse dense du pic
            A3s = A3 * mask
            A3s = A3s / A3s.sum(-1, keepdim=True).clamp(min=1e-9)
            ms = A3s.gather(2, pki).squeeze(-1)                # masse creuse du pic
            ACC["md"] += float(md[:, QPOS].mean())
            ACC["ms"] += float(ms[:, QPOS].mean())
            ACC["nl"] += 1
            A = A3s.unsqueeze(0)
            del mask, B, A3s
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

log = OUT / "gpt2_masse_pic.txt"
log.write_text(f"{MID} | banc induction | frac=0,375 | masse du pic : dense vs creuse\n"
               "dp  kagg  bagg  perte      Delta    pic%   masse_pic_dense  masse_pic_creuse\n",
               encoding="utf-8")
D = 0.0


def run(dp, kagg, bagg):
    global D
    if dp is None:
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
    else:
        CFG["mode"] = "pca"; CFG["dp"] = dp; CFG["kagg"] = kagg; CFG["bagg"] = bagg
        CFG["W"] = 32; CFG["m"] = 16
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
            f.write(f"---  dense  ---  {l4:9.5f}  ---      ---    ---     ---\n")
        print(f"dense: {l4:.5f}", flush=True)
        return l4
    nl = max(ACC["nl"], 1)
    pk = 100.0 * ACC["peak"] / max(ACC["peakn"], 1)
    md, ms = ACC["md"] / nl, ACC["ms"] / nl
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{dp:3d}  {kagg:4s}  {bagg:4s}  {l4:9.5f}  {l4 - D:+.5f}  {pk:5.1f}  "
                f"{md:14.4f}  {ms:15.4f}\n")
    print(f"dp={dp} {kagg}/{bagg}: perte={l4:.5f} Delta={l4-D:+.5f} pic={pk:.1f}% "
          f"masse_pic dense={md:.4f} creuse={ms:.4f}", flush=True)
    return l4


D = run(None, None, None)
run(8, "max", "max")
run(8, "sum", "max")
run(8, "sum", "sum")
run(16, "sum", "max")
run(32, "sum", "max")
run(32, "max", "max")
print("FIN")
