# -*- coding: utf-8 -*-
"""L'ordre des selecteurs sur Qwen2.5-0.5B tient-il EN DOMAINE et a contexte plus long ?
Wikitext, T=1024, W=256, Lb=64, m=2 (12 candidats). Variantes : dense | m0 | cle moyenne |
aleatoire (1 graine) | oracle LSE. Comparer a la run corpus T=512 (oracle 0,2886 le meilleur)."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "Qwen/Qwen2.5-0.5B"
T, Lb, W, M = 1024, 64, 256, 2
CFG = {"mode": "dense", "sel": "mean", "seed": 1234}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
buf = ""
for r in ds:
    buf += r["text"]
    if len(buf) > 200000:
        break
ids = tok(buf, return_tensors="pt", truncation=True, max_length=T).input_ids
NB = T // Lb
log = OUT / "invariant_qwen_domaine.txt"
log.write_text(f"{MID} | wikitext | T={T} | W={W} | Lb={Lb} | m={M} | candidats={NB}\n",
               encoding="utf-8")


def run(tag):
    with torch.no_grad():
        out = model(ids)
        lm = float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none").mean())
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{tag:18s} perte={lm:.4f} (p={math.exp(lm):5.3f})\n")
        f.flush()
    return lm


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
        if CFG["mode"] == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(g > 1))
        else:
            ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
            nb = qn // Lb
            j = torch.arange(qn); p = torch.arange(qn)
            allowed = (j[None, :] >= (p - W + 1).clamp(min=0)[:, None])[None].expand(H, qn, qn).clone()
            bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
            if CFG["mode"] != "m0":
                if CFG["sel"] == "oracle":
                    S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                    Sb = S[:, :, :nb * Lb].view(H, qn, nb, Lb)
                    caus = (torch.arange(nb * Lb)[None, :] <= p[:, None]).view(qn, nb, Lb)
                    Sb = Sb.masked_fill((~caus)[None], float("-inf"))
                    lse = torch.logsumexp(Sb, dim=3).masked_fill(~bv.T[None], float("-inf"))
                    _, ti = torch.topk(lse, M, dim=2)
                elif CFG["sel"] == "rand":
                    gen = torch.Generator().manual_seed(CFG["seed"])
                    ti = torch.randint(0, nb, (H, qn, M), generator=gen)
                    ok = bv.T[None].expand(H, qn, nb).gather(2, ti.clamp(max=nb - 1))
                    ti = torch.where(ok, ti, torch.full_like(ti, -1))
                else:
                    mu = ke[0].view(nb, Lb, H, D).mean(1)
                    sc = torch.einsum("hqd,bhd->hqb", q[0], mu) / math.sqrt(D)
                    sc = sc.masked_fill((~bv).T[None], float("-inf"))
                    tv, ti = torch.topk(sc, M, dim=2)
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

run("dense")
CFG.update({"mode": "m0"}); run("m0_fenetre")
CFG.update({"mode": "sparse", "sel": "mean"}); run("mean_cle_moyenne")
CFG.update({"sel": "rand"}); run("rand_seed1234")
CFG.update({"sel": "oracle"}); run("oracle_LSE")
print("ok")
