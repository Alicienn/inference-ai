# -*- coding: utf-8 -*-
"""Diagnostic cible : pour la couche 0, requete p=300, comparer pas a pas
selection top-m, ensemble autorise, et sortie d'attention brute (avant o_proj)."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M, P0 = 512, 32, 128, 2, 300
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
IDS = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0][:T][None]
CAP = {}


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
        nb = qn // Lb
        j = torch.arange(qn); p = torch.arange(qn)
        causal = (j[None, :] <= p[:, None])
        allowed = ((j[None, :] >= (p - W + 1).clamp(min=0)[:, None]) & causal)[None].expand(H, qn, qn).clone()
        bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
        Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
        Sb = Sp[:, :, :nb * Lb].view(H, qn, nb, Lb)
        causb = causal[:, :nb * Lb].view(qn, nb, Lb)
        Sb = Sb.masked_fill((~causb)[None], float("-inf"))
        sc = Sb.max(dim=3).values.masked_fill((~bv).T[None], float("-inf"))
        tv, ti = torch.topk(sc, M, dim=2)
        ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
        onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
        _vv = ti >= 0
        _hh, _pp, _ = torch.nonzero(_vv, as_tuple=True)
        onehot[_hh, _pp, ti[_vv]] = True
        allowed |= onehot[:, :, (j // Lb)] & causal[None]
        o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        if li == 0:
            CAP["in"] = hidden_states.detach(); CAP["pos"] = (cos.detach(), sin.detach())
            CAP["o"] = o.detach(); CAP["ti"] = ti[:, P0].clone()
            CAP["row"] = allowed[0, P0].clone(); CAP["sc"] = sc[:, P0].clone()
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))
with torch.no_grad():
    _ = model(IDS)

att = model.model.layers[0].self_attn
h_in = CAP["in"]; cos, sin = CAP["pos"]
q = att.q_proj(h_in).view(1, T, H, D).transpose(1, 2)
k = att.k_proj(h_in).view(1, T, Hkv, D).transpose(1, 2)
v = att.v_proj(h_in).view(1, T, Hkv, D).transpose(1, 2)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
g = att.num_key_value_groups
ke = k.repeat_interleave(g, 1)[0]; ve = v.repeat_interleave(g, 1)[0]
q0 = q[0]
nb = T // Lb
p = P0; lo = max(0, p - W + 1)
blk = [b for b in range(nb) if (b + 1) * Lb <= lo]
sc_ref = torch.stack([(q0[:, p].unsqueeze(1) * ke[:, b * Lb:(b + 1) * Lb]).sum(-1).max(-1).values
                      for b in blk], dim=1) / math.sqrt(D)
top_ref = sc_ref.topk(M, dim=1).indices
sel_ref = [[blk[t] for t in top_ref[h].tolist()] for h in range(H)]
sel_pat = [[int(x) for x in CAP["ti"][h].tolist()] for h in range(H)]
print("blocs candidats :", blk)
print("selection patchee (tete 0-3) :", sel_pat[:4])
print("selection reference (tete 0-3) :", sel_ref[:4])
print("scores pataches (tete 0, blocs valides) :", [round(float(x), 4) for x in CAP["sc"][0][blk]])
print("scores reference (tete 0) :", [round(float(x), 4) for x in sc_ref[0]])
o_ref = torch.zeros(H, T, D)
for h in range(H):
    allowed = set(range(lo, p + 1))
    for b in sel_ref[h]:
        allowed |= set(range(b * Lb, (b + 1) * Lb))
    idx = torch.tensor(sorted(allowed), dtype=torch.long)
    lg = (q0[h, p] @ ke[h, idx].T) / math.sqrt(D)
    o_ref[h, p] = torch.softmax(lg, dim=0) @ ve[h, idx]
print("nb positions autorisees : patche", int(CAP["row"].sum()), "| reference", len(allowed))
print("ecart sortie brute p=300 (tete 0) :", float((o_ref[0, p] - CAP["o"][0, 0, p]).abs().max()))
print("ecart sortie brute p=300 (toutes tetes) :", float((o_ref[:, p] - CAP["o"][0, :, p]).abs().max()))
print("norme sortie brute p=300 :", float(CAP["o"][0, :, p].abs().max()))
