# -*- coding: utf-8 -*-
"""CORRECTION DU BUG CAUSAL : les masques creux precedents n'imposaient pas j <= p
(fuite du futur). On ajoute la borne causale et on RE-MESURE les deux modeles.
T=512, W=128, Lb=64 (6 blocs candidats), m=2, corpus local. dense | m0 | cle moyenne |
aleatoire (3 graines) | oracle LSE."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
T, Lb, W, M = 512, 64, 128, 2
CFG = {"mode": "dense", "sel": "mean", "seed": 0}
log = OUT / "CAUSAL_CORRIGE.txt"
log.write_text("CORRECTION BUG CAUSAL (j <= p impose) | T=512 W=128 Lb=64 m=2 | corpus local\n",
               encoding="utf-8")


def patched(H, D):
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
            causal = (j[None, :] <= p[:, None])
            allowed = ((j[None, :] >= (p - W + 1).clamp(min=0)[:, None]) & causal)[None].expand(H, qn, qn).clone()
            bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
            if CFG["mode"] != "m0":
                if CFG["sel"] == "oracle":
                    S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                    Sb = S[:, :, :nb * Lb].view(H, qn, nb, Lb)
                    causb = causal[:, :nb * Lb].view(qn, nb, Lb)
                    Sb = Sb.masked_fill((~causb)[None], float("-inf"))
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
                allowed |= onehot[:, :, (j // Lb)] & causal[None]
            o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"

for MID, mod in (("HuggingFaceTB/SmolLM2-135M", "llama"), ("Qwen/Qwen2.5-0.5B", "qwen2")):
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    cfg = model.config
    H = cfg.num_attention_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    if mod == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    else:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    ids = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0, :T][None]
    for layer in model.model.layers:
        layer.self_attn.forward = patched(H, D).__get__(layer.self_attn, type(layer.self_attn))

    def run(tag):
        with torch.no_grad():
            out = model(ids)
            lm = float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none").mean())
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{MID.split('/')[-1]:16s} {tag:18s} perte={lm:.4f} (p={math.exp(lm):7.3f})\n")
            f.flush()
        return lm

    CFG.update({"mode": "dense"}); run("dense")
    CFG.update({"mode": "m0"}); run("m0_fenetre")
    CFG.update({"mode": "sparse", "sel": "mean"}); lmean = run("mean_cle_moyenne")
    CFG.update({"sel": "oracle"}); lor = run("oracle_LSE")
    CFG.update({"sel": "rand"})
    rands = []
    for sd in (1000, 2000, 3000):
        CFG["seed"] = sd
        rands.append(run(f"rand_seed{sd}"))
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{MID.split('/')[-1]:16s} ALEATOIRE moy={statistics.mean(rands):.4f} "
                f"min={min(rands):.4f} max={max(rands):.4f} | mean bat hasard ? "
                f"{lmean < min(rands)} | oracle bat hasard ? {lor < min(rands)}\n\n")
        f.flush()
print("ok")
