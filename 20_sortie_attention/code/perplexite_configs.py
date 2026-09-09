# -*- coding: utf-8 -*-
"""Separer l'effet FENETRE de l'effet BLOCS, et tester si le selecteur bat le HASARD.
SmolLM2-135M, T=1024, Lb=64, corpus local. Variantes :
  dense | W256 m0 (fenetre seule) | W256 m4 | W256 m8 | W256 m4 ALEATOIRE | W512 m4
Harnais identique a perplexite_creuse.py (dense patche = perte native, valide)."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb = 1024, 64
CFG = {"mode": "dense", "W": 256, "m": 0, "sel": "mean", "seed": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ids = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0, :T][None]

log = OUT / "perplexite_configs.txt"
log.write_text(f"SmolLM2-135M | T={T} | Lb={Lb} | corpus local | tokens={T}\n", encoding="utf-8")


def run(tag):
    with torch.no_grad():
        out = model(ids)
        ls = F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")
        lm, lw = float(ls.mean()), float(ls[CFG["W"]:].mean())
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{tag:22s} perte={lm:.4f} (p={math.exp(lm):6.2f}) | hors fenetre={lw:.4f} "
                f"(p={math.exp(lw):6.2f})\n")
        f.flush()


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
        W, m = CFG["W"], CFG["m"]
        if CFG["mode"] == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(g > 1))
        else:
            ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
            nb = qn // Lb
            p = torch.arange(qn)
            j = torch.arange(qn)
            allowed = (j[None, :] >= (p - W + 1).clamp(min=0)[:, None])[None].expand(H, qn, qn).clone()
            if m > 0:
                mu = ke[0].view(nb, Lb, H, D).mean(1)
                sc = torch.einsum("hqd,bhd->hqb", q[0], mu) / math.sqrt(D)
                bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
                sc = sc.masked_fill((~bv).T[None], float("-inf"))
                if CFG["sel"] == "rand":
                    gsel = torch.Generator().manual_seed(CFG["seed"])
                    ti = torch.randint(0, nb, (H, qn, m), generator=gsel)
                    ti = torch.where(bv.T[None].expand(H, qn, nb).gather(2, ti.clamp(max=nb - 1)),
                                     ti, torch.full_like(ti, -1))
                else:
                    tv, ti = torch.topk(sc, m, dim=2)
                    ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
                onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
                _vv = ti >= 0
                _hh, _pp, _ = torch.nonzero(_vv, as_tuple=True)
                onehot[_hh, _pp, ti[_vv]] = True
                allowed |= onehot[:, :, (j // Lb)]
            o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = patched().__get__(layer.self_attn, type(layer.self_attn))

RUNS = [("dense", "dense", 1024, 0, "mean", 0),
        ("W256_m0", "sparse", 256, 0, "mean", 0),
        ("W256_m4", "sparse", 256, 4, "mean", 0),
        ("W256_m8", "sparse", 256, 8, "mean", 0),
        ("W256_m4_RANDOM", "sparse", 256, 4, "rand", 1234),
        ("W512_m4", "sparse", 512, 4, "mean", 0)]
for tag, mode, W, m, sel, seed in RUNS:
    CFG.update({"mode": mode, "W": W, "m": m, "sel": sel, "seed": seed})
    run(tag)
print("ok")
