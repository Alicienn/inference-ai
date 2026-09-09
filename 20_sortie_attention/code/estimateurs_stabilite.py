# -*- coding: utf-8 -*-
"""Stabilite de 'maxip >= oracle' : (A) replication sur Qwen2.5-0.5B ;
(B) balayage de m sur SmolLM2. Harnais causal corrige, auto-test inclus."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
T, Lb, W = 512, 32, 128
CFG = {"mode": "dense", "sel": "mean", "seed": 0, "M": 2}
log = OUT / "estimateurs_stabilite.txt"
log.write_text("Stabilite maxip | T=512 W=128 Lb=32 | harnais causal corrige\n", encoding="utf-8")
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"


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
        M = CFG["M"]
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
                sel = CFG["sel"]
                if sel == "rand":
                    gen = torch.Generator().manual_seed(CFG["seed"])
                    ti = torch.randint(0, nb, (H, qn, M), generator=gen)
                    ok = bv.T[None].expand(H, qn, nb).gather(2, ti.clamp(max=nb - 1))
                    ti = torch.where(ok, ti, torch.full_like(ti, -1))
                else:
                    S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                    Sb = S[:, :, :nb * Lb].view(H, qn, nb, Lb)
                    causb = causal[:, :nb * Lb].view(qn, nb, Lb)
                    Sb = Sb.masked_fill((~causb)[None], float("-inf"))
                    if sel == "oracle":
                        sc = torch.logsumexp(Sb, dim=3)
                    elif sel == "maxip":
                        sc = Sb.max(dim=3).values
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


for MID, mod in (("Qwen/Qwen2.5-0.5B", "qwen2"), ("HuggingFaceTB/SmolLM2-135M", "llama")):
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
    IDS = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0, :T][None]
    for layer in model.model.layers:
        layer.self_attn.forward = patched(H, D).__get__(layer.self_attn, type(layer.self_attn))

    def loss(ids=IDS):
        with torch.no_grad():
            out = model(ids)
            return F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")

    CFG.update({"mode": "m0"})
    d = float((loss(IDS)[:255] - loss(IDS[:, :256])).abs().max())
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n== {MID.split('/')[-1]} == AUTO-TEST CAUSAL {d:.3e} "
                f"{'OK' if d < 1e-4 else 'ECHEC'}\n")
    if mod == "qwen2":
        plan = [("dense", "dense", 2), ("m0", "m0", 2), ("maxip", "maxip", 2),
                ("oracle", "oracle", 2), ("meankey", "meankey", 2),
                ("rand1000", "rand", 2), ("rand2000", "rand", 2)]
    else:
        plan = [("m1_maxip", "maxip", 1), ("m1_oracle", "oracle", 1),
                ("m4_maxip", "maxip", 4), ("m4_oracle", "oracle", 4),
                ("m8_maxip", "maxip", 8), ("m8_oracle", "oracle", 8),
                ("m4_rand", "rand", 4)]
    for tag, sel, mm in plan:
        CFG.update({"mode": "dense" if tag == "dense" else ("m0" if tag == "m0" else "sparse"),
                    "sel": sel, "M": mm, "seed": 1000 if "1000" in tag else (2000 if "2000" in tag else 0)})
        v = float(loss().mean())
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{tag:12s} perte={v:.4f} (p={math.exp(v):8.3f})\n")
            f.flush()
print("ok")
