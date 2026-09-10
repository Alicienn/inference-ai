# -*- coding: utf-8 -*-
"""INDEX DEPLOYABLE vs ORACLE a contexte long. Le creux bat-il encore le dense
quand la selection n'est plus oracle mais l'index PCA causal deployable
(base = 256 premiers tokens, score de bloc = max sur 8 projections) ?
T=2048 (1 tranche) et T=1024 (2 tranches), fraction lue 37,5 %, Lb=4."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
PMIN, NBASE, DP = 320, 256, 8
CFG = {"mode": "dense", "Lb": 4, "W": 256, "m": 128}
BASE = {}

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
            K = ke[0].float()
            Kc = K - K.mean(1, keepdim=True)
            U, _, _ = torch.linalg.svd(Kc.transpose(1, 2), full_matrices=False)
            BASE[li] = U[:, :, :DP].contiguous()
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2)
        del S
        Lb, W, m = CFG["Lb"], CFG["W"], CFG["m"]
        if CFG["mode"] in ("oracle", "pca") and qn >= W + Lb + 1:
            nb = qn // Lb
            if CFG["mode"] == "oracle":
                B = A[:, :, :nb * Lb].view(h, qn, nb, Lb).amax(-1)
            else:
                U = BASE[li]
                qp = torch.einsum("hqd,hde->hqe", q[0], U)
                kp = torch.einsum("hkd,hde->hke", ke[0], U)
                sc = qp[:, :, None, :] * kp[:, None, :, :]      # (h, q, k, e)
                sc = sc.amax(-1)                                 # max sur projections
                B = sc[:, :, :nb * Lb].view(h, qn, nb, Lb).amax(-1)
                del qp, kp, sc
            bid = torch.arange(nb)
            elig = (bid[None, :] * Lb + Lb - 1) <= (torch.arange(qn) - W)[:, None]
            _, sel = torch.topk(B * elig[None], min(m, nb), dim=2)
            keys = (sel[..., None] * Lb + torch.arange(Lb)).reshape(h, qn, -1)
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
print(f"base collectee sur {len(BASE)} couches", flush=True)


def run(T, nchunks, mode):
    CFG["mode"] = mode
    CFG["W"] = T // 8
    CFG["m"] = (T // 4) // CFG["Lb"]
    tot, n = 0.0, 0
    for c in range(nchunks):
        ids = ALL[c * T:(c + 1) * T][None]
        with torch.no_grad():
            logits = model(ids).logits[0]
        ls = F.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
        tot += float(ls[PMIN:].sum()); n += int(ls[PMIN:].numel())
    return tot / n


log = OUT / "index_vs_oracle.txt"
log.write_text(f"{MID} | Lb=4 | fraction 37,5 % (W=T/8, m*Lb=T/4) | base PCA={DP} "
               f"sur {NBASE} tokens | p>={PMIN}\n"
               "T      config      perte       Delta\n", encoding="utf-8")
res = {}
for T, nch in ((1024, 2), (2048, 1)):
    d = run(T, nch, "dense")
    o = run(T, nch, "oracle")
    p = run(T, nch, "pca")
    res[T] = (d, o, p)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{T:5d}  dense   {d:10.5f}   ---\n")
        f.write(f"{T:5d}  oracle  {o:10.5f}   {o-d:+.5f}\n")
        f.write(f"{T:5d}  pca     {p:10.5f}   {p-d:+.5f}\n")
    print(f"T={T} dense={d:.5f} oracle={o:.5f}({o-d:+.5f}) pca={p:.5f}({p-d:+.5f})", flush=True)
print("FIN")

