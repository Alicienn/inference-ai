# -*- coding: utf-8 -*-
"""CONTROLE D'INDUCTION : le gain de perte sur texte continu est-il un artefact ?
On place une aiguille de 16 tokens en position 50 et la meme a 496 ; la perte sur
les 15 derniers tokens exige d'aller chercher le bloc de l'aiguille (position 50).
Si l'index deployable rate ce bloc, la perte explose la ou le dense reussit."""
import math, pathlib, torch, torch.nn.functional as F, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, PMIN, NBASE, DP, LB, NT = 512, 40, 256, 8, 4, 8
CFG = {"mode": "dense", "W": 64, "m": 32}
BASE = {}
FRACS = [0.375, 0.125, 0.0625]

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]


def make_fwd(li):
    def fwd(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
        bsz, qn, _ = hidden_states.shape
        sh = (bsz, qn, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(sh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(sh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(sh).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        g = self.num_key_value_groups
        ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
        h = q.shape[1]
        if CFG["mode"] == "collect":
            K = ke[0].float(); Kc = K - K.mean(1, keepdim=True)
            U, _, _ = torch.linalg.svd(Kc.transpose(1, 2), full_matrices=False)
            BASE[li] = U[:, :, :DP].contiguous()
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2)
        del S
        W, m = CFG["W"], CFG["m"]
        if CFG["mode"] == "pca" and m > 0:
            nb = qn // LB
            U = BASE[li]
            qp = torch.einsum("hqd,hde->hqe", q[0], U)
            kp = torch.einsum("hkd,hde->hke", ke[0], U)
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
            A = A * mask
            A = A / A.sum(-1, keepdim=True).clamp(min=1e-9)
            del mask, B
        o = torch.einsum("hqk,hkd->hqd", A, ve[0])
        del A
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))

CFG["mode"] = "collect"
with torch.no_grad():
    model(ALL[:NBASE][None])
print("base collectee", flush=True)

# sequences a aiguille
seqs = []
rng = random.Random(7)
for i in range(NT):
    s = rng.randint(2000, len(ALL) - T - 32)
    needle = ALL[s:s + 16].clone()
    seq = ALL[0:T].clone()
    seq[50:66] = needle
    seq[T - 16:T] = needle
    seqs.append(seq)

POS = list(range(T - 16, T - 1))     # 15 tokens dont la prediction exige l'aiguille


def run(mode, frac):
    if mode == "dense":
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
    else:
        CFG["mode"] = "pca"
        CFG["W"] = max(8, int(T * frac / 3))
        CFG["m"] = max(1, int((T * frac * 2 / 3) // LB))
    tot, n = 0.0, 0
    for seq in seqs:
        ids = seq[None]
        with torch.no_grad():
            logits = model(ids).logits[0]
        ls = F.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
        tot += float(ls[POS].sum()); n += len(POS)
    return tot / n, CFG["W"], CFG["m"]


log = OUT / "induction.txt"
log.write_text(f"{MID} | T={T} aiguille 16 tokens en pos 50 et {T-16} | perte sur les "
               f"{len(POS)} derniers tokens (exige l'aiguille) | index PCA d'={DP} max\n"
               "config        W    m    perte_aiguille\n", encoding="utf-8")
d, _, _ = run("dense", None)
with log.open("a", encoding="utf-8") as f:
    f.write(f"dense        ---  ---  {d:12.5f}\n")
print(f"dense perte_aiguille={d:.5f}", flush=True)
for fr in FRACS:
    p, W, m = run("pca", fr)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"pca {fr:.3f}    {W:4d} {m:4d}  {p:12.5f}  (Delta {p-d:+.5f})\n")
    print(f"frac={fr:.3f} W={W} m={m} perte={p:.5f} Delta={p-d:+.5f}", flush=True)
print("FIN")

