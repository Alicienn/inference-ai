# -*- coding: utf-8 -*-
"""Trancher : le selecteur par cle moyenne est-il PIRE que le hasard, ou dans sa distribution ?
Et un ORACLE de selection (top-m blocs par LSE exact) aiderait-il ?
SmolLM2-135M, T=512, W=128, Lb=64, m=2 (6 blocs candidats), corpus local.
Harnais valide (dense patche = perte native)."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M = 512, 64, 128, 2
CFG = {"mode": "dense", "sel": "mean", "seed": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H, KVH, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ids = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0, :T][None]

log = OUT / "perplexite_hasard.txt"
log.write_text(f"SmolLM2-135M | T={T} | W={W} | Lb={Lb} | m={M} | candidats={6}\n", encoding="utf-8")


def run(tag):
    with torch.no_grad():
        out = model(ids)
        ls = F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")
        lm = float(ls.mean())
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{tag:16s} perte={lm:.4f} (p={math.exp(lm):6.2f})\n")
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
            p = torch.arange(qn); j = torch.arange(qn)
            allowed = (j[None, :] >= (p - W + 1).clamp(min=0)[:, None])[None].expand(H, qn, qn).clone()
            bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
            if M > 0 and CFG["mode"] != "m0":
                if CFG["sel"] == "oracle":
                    S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                    Sb = S[:, :, :nb * Lb].view(H, qn, nb, Lb)
                    caus = (torch.arange(nb * Lb)[None, :] <= p[:, None])
                    Sb = Sb.masked_fill((~caus).view(qn, nb, Lb)[None], float("-inf"))
                    lse = torch.logsumexp(Sb, dim=3)
                    lse = lse.masked_fill(~bv.T[None], float("-inf"))
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

dense = run("dense")
CFG.update({"mode": "m0", "sel": "mean"}); run("m0_fenetre")
CFG.update({"mode": "sparse", "sel": "mean"}); mean_l = run("mean_cle_moyenne")
CFG.update({"sel": "oracle"}); oracle_l = run("oracle_LSE")
CFG.update({"sel": "rand"})
rands = []
for seed in range(12):
    CFG["seed"] = 1000 + seed
    rands.append(run(f"rand_seed{seed}"))
import statistics
with log.open("a", encoding="utf-8") as f:
    f.write(f"\nALEATOIRE (12 tirages) : moyenne={statistics.mean(rands):.4f} "
            f"ecart-type={statistics.pstdev(rands):.4f} min={min(rands):.4f} max={max(rands):.4f}\n")
    f.write(f"SELECTEUR cle moyenne = {mean_l:.4f} | position dans la distribution : "
            f"{'DANS' if min(rands) <= mean_l <= max(rands) else 'EN DEHORS'} "
            f"({sum(1 for r in rands if r < mean_l)}/12 tirages meilleurs)\n")
    f.write(f"ORACLE LSE = {oracle_l:.4f} | dense = {dense:.4f}\n")
print("ok")
