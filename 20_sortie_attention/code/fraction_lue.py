# -*- coding: utf-8 -*-
"""FRONTIERE DE FRACTION LUE : ou le creux cesse-t-il de battre le dense ?
Index PCA causal deployable + agregation max. Lb=4. Fenetre = 1/3 du budget lu,
blocs = 2/3 (proportion du reglage de tete W=T/8, m*Lb=T/4)."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
PMIN, NBASE, DP, LB = 320, 256, 8, 4
CFG = {"mode": "dense", "W": 128, "m": 64}
BASE = {}
FRACS = [0.375, 0.25, 0.125, 0.0625]

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
        if CFG["mode"] == "pca" and m > 0 and qn >= W + LB + 1:
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


def run(T, nchunks, frac):
    if frac is None:
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
    else:
        CFG["mode"] = "pca"
        CFG["W"] = max(8, int(T * frac / 3))
        CFG["m"] = max(1, int((T * frac * 2 / 3) // LB))
    tot, n = 0.0, 0
    for c in range(nchunks):
        ids = ALL[c * T:(c + 1) * T][None]
        with torch.no_grad():
            logits = model(ids).logits[0]
        ls = F.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
        tot += float(ls[PMIN:].sum()); n += int(ls[PMIN:].numel())
    return tot / n, CFG["W"], CFG["m"]


log = OUT / "fraction_lue.txt"
log.write_text(f"{MID} | Lb={LB} | index PCA d'={DP} base {NBASE} | agregation max | p>={PMIN}\n"
               "T      frac   W     m     clefs_lues  perte      Delta\n", encoding="utf-8")
for T, nch in ((1024, 2), (2048, 1)):
    d, _, _ = run(T, nch, None)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{T:5d}  dense  ---   ---   ---         {d:10.5f}  ---\n")
    print(f"T={T} dense={d:.5f}", flush=True)
    for fr in FRACS:
        p, W, m = run(T, nch, fr)
        cles = W + m * LB
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{T:5d}  {fr:.3f}  {W:4d}  {m:4d}  {cles:6d} ({cles/T:.3f})  "
                    f"{p:10.5f}  {p-d:+.5f}\n")
        print(f"T={T} frac={fr:.3f} W={W} m={m} cles={cles}({cles/T:.3f}) "
              f"perte={p:.5f} Delta={p-d:+.5f}", flush=True)
print("FIN")
