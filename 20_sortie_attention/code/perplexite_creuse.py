# -*- coding: utf-8 -*-
"""Le forcage du bloc du sink change-t-il la PERTE au niveau modele ? (v2, o_proj corrige)
Patch attention SmolLM2-135M : dense / creux (fenetre W + m blocs de 64 par cle moyenne)
/ creux + bloc 0 force. Teacher forcing, T=1024, corpus local, validation native incluse."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W = 1024, 64, 256
MODE = {"m": "dense", "mm": 8}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ids = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0, :T][None]

log = OUT / "perplexite_creuse.txt"
log.write_text(f"SmolLM2-135M | T={T} | W={W} | Lb={Lb} | tokens evalues={T}\n", encoding="utf-8")


def run(tag):
    with torch.no_grad():
        out = model(ids)
        ls = F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")
        lm, lw = float(ls.mean()), float(ls[W:].mean())
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{tag:14s} perte={lm:.4f} (p={math.exp(lm):.2f}) | hors fenetre={lw:.4f} "
                f"(p={math.exp(lw):.2f})\n")
        f.flush()
    return lm


run("native")

def patched():
    def fwd(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
        bsz, qn, _ = hidden_states.shape
        sh = (bsz, qn, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(sh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(sh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(sh).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        g = self.num_key_value_groups
        if MODE["m"] == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(g > 1))
        else:
            ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
            nb = qn // Lb
            mu = ke[0].view(nb, Lb, H, D).mean(1)
            sc = torch.einsum("hqd,bhd->hqb", q[0], mu) / math.sqrt(D)
            p = torch.arange(qn)
            bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
            sc = sc.masked_fill((~bv).T[None], float("-inf"))
            tv, ti = torch.topk(sc, MODE["mm"], dim=2)
            ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
            if MODE["m"] == "mean_sink":
                ti[:, :, 0] = 0
            j = torch.arange(qn)
            onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
            _vv = ti >= 0
            _hh, _pp, _ = torch.nonzero(_vv, as_tuple=True)
            onehot[_hh, _pp, ti[_vv]] = True
            allowed = onehot[:, :, (j // Lb)] | (j[None, :] >= (p - W + 1).clamp(min=0)[:, None])[None]
            o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = patched().__get__(layer.self_attn, type(layer.self_attn))

for name, mm in (("dense", 8), ("mean", 8), ("mean_sink", 8), ("mean", 4), ("mean_sink", 4)):
    MODE["m"], MODE["mm"] = name, mm
    run(f"{name}_m{mm}")
print("ok")
