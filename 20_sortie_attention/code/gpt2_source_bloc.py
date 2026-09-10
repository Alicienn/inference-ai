# -*- coding: utf-8 -*-
"""TEST DECISIF SUR GPT-2 : l'index trouve-t-il le bloc SOURCE DISTANT ?
Banc valide (induction +2,95 nat, tete d'induction pure couche 11).
Configs : dense | oracle (masse d'attention vraie) | index PCA deployable.
Fracs 0,375 et 0,125. Lb=4, W=T/8, m*Lb=T/4 (ou fraction equivalente)."""
import math, pathlib, torch, torch.nn.functional as F, random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "gpt2"
T, NBASE, DP, LB, NT, SPAN = 256, 128, 8, 4, 4, 64
QPOS = list(range(193, 256))
CFG = {"mode": "dense", "W": 32, "m": 16, "dbg": True}
BASE = {}
ACC = {"sel": 0, "n": 0, "nl": 0, "m1": 0.0, "peak": 0, "peakn": 0}

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
            BASE[li] = U[:, :, :DP].contiguous()
        S = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        j = torch.arange(qn)
        S = S.masked_fill(j[None, None, :] > j[None, :, None], float("-inf"))
        A = torch.softmax(S, dim=-1)
        del S
        W, m = CFG["W"], CFG["m"]
        if CFG["mode"] in ("oracle", "pca") and m > 0 and qn == T:
            nb = qn // LB
            if CFG["mode"] == "oracle":
                B = A[:, :, :nb * LB].view(h, qn, nb, LB).sum(-1)
            else:
                U = BASE[li]
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
            ACC["sel"] += int(mask[:, QPOS, 0:SPAN].any(-1).sum())
            ACC["n"] += mask[:, QPOS, 0:SPAN].any(-1).numel(); ACC["nl"] += 1
            pk = A.argmax(-1)
            pkb = pk // LB
            hit = (sel == pkb[..., None]).any(-1)
            inwin = pk > (torch.arange(qn)[None, :] - W)
            keep = hit | inwin
            if CFG.get("dbg"):
                print("DBG", "h", h, "qn", qn, "m", m, "A", tuple(A.shape),
                      "sel", tuple(sel.shape), "pkb", tuple(pkb.shape),
                      "hit", tuple(hit.shape), "keep", tuple(keep.shape), flush=True)
                CFG["dbg"] = False
            ACC["peak"] += int(keep[:, QPOS].sum())
            ACC["peakn"] += keep[:, QPOS].numel()
            A = A * mask
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
print("base collectee sur", len(BASE), "couches", flush=True)

seqs = []
rng = random.Random(11)
for i in range(NT):
    s = rng.randint(2000, len(ALL) - SPAN - 8)
    seqs.append(ALL[s:s + SPAN].clone().repeat(4))

log = OUT / "gpt2_pic_bloc.txt"
log.write_text(f"{MID} | span 64 tok x4, T={T} | Lb={LB} base PCA d'={DP} sur {NBASE} tok\n"
               "config        perte_copie4  Delta_vs_dense  bloc_source_sel%  bloc_du_PIC%\n", encoding="utf-8")


D = 0.0


def run(mode, frac):
    if mode == "dense":
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
    else:
        CFG["mode"] = mode
        CFG["W"] = max(8, int(T * frac / 3)); CFG["m"] = max(1, int((T * frac * 2 / 3) // LB))
        for k in ACC: ACC[k] = 0
    l4 = 0.0
    for seq in seqs:
        with torch.no_grad():
            logits = model(seq[None]).logits[0]
        ls = F.cross_entropy(logits[:-1], seq[1:], reduction="none")
        l4 += float(ls[193:T - 1].mean())
    l4 /= NT
    tag = "dense" if mode == "dense" else f"{mode} {frac:.3f}"
    selp = 100.0 * ACC["sel"] / max(ACC["n"], 1) if mode != "dense" else float("nan")
    peakp = 100.0 * ACC["peak"] / max(ACC["peakn"], 1) if mode != "dense" else float("nan")
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{tag:12s} {l4:12.5f}  {l4 - D:+.5f}      {selp:6.1f}         {peakp:6.1f}\n")
    print(f"{tag}: perte_copie4={l4:.5f} (Delta {l4-D:+.5f}) bloc_source_sel={selp:.1f}% bloc_du_PIC={peakp:.1f}%", flush=True)
    return l4


D = run("dense", None)
run("oracle", 0.375)
run("pca", 0.375)
run("pca", 0.125)
print("FIN")

