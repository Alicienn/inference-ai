# -*- coding: utf-8 -*-
"""La statistique heavy-hitter est-elle compressible en un sketch compact ?
Par bloc, on stocke les r cles de plus grande NORME (r=1,2,4,8 ; 3 a 25 % des octets de cles)
et on score par max du produit interne sur ce sous-ensemble. Harnais causal corrige."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M = 512, 32, 128, 2
CFG = {"mode": "dense", "sel": "maxip", "r": 2, "seed": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
IDS = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0, :T][None]
NB = T // Lb
log = OUT / "sketch_heavy.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} | sketch = top-r cles par norme\n",
               encoding="utf-8")


def loss(ids=IDS):
    with torch.no_grad():
        out = model(ids)
        return F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")


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
                    if sel == "sketch":
                        r = CFG["r"]
                        kn = ke[0].norm(dim=2).view(H, nb, Lb)
                        _, idx = torch.topk(kn, r, dim=2)                 # (H, nb, r)
                        pos = (torch.arange(nb).view(1, nb, 1) * Lb + idx).reshape(H, nb * r)
                        Sg = S.gather(2, pos[:, None, :].expand(H, qn, nb * r)).view(H, qn, nb, r)
                        sc = Sg.max(dim=3).values
                    else:
                        Sb = S[:, :, :nb * Lb].view(H, qn, nb, Lb)
                        causb = causal[:, :nb * Lb].view(qn, nb, Lb)
                        Sb = Sb.masked_fill((~causb)[None], float("-inf"))
                        sc = Sb.max(dim=3).values if sel == "maxip" else torch.logsumexp(Sb, dim=3)
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


for layer in model.model.layers:
    layer.self_attn.forward = patched().__get__(layer.self_attn, type(layer.self_attn))

CFG.update({"mode": "m0"})
d = float((loss(IDS)[:255] - loss(IDS[:, :256])).abs().max())
with log.open("a", encoding="utf-8") as f:
    f.write(f"AUTO-TEST CAUSAL {d:.3e} -> {'OK' if d < 1e-3 else 'ECHEC'}\n")

res = {}
CFG.update({"mode": "dense"}); res["dense"] = float(loss().mean())
CFG.update({"mode": "m0"}); res["m0_fenetre"] = float(loss().mean())
CFG.update({"mode": "sparse", "sel": "maxip"}); res["maxip_toutes_cles"] = float(loss().mean())
CFG.update({"sel": "oracle"}); res["oracle_LSE"] = float(loss().mean())
for r in (1, 2, 4, 8):
    CFG.update({"sel": "sketch", "r": r}); res[f"sketch_r{r}"] = float(loss().mean())
CFG.update({"sel": "rand"})
rands = []
for sd in (1000, 2000):
    CFG["seed"] = sd
    rands.append(float(loss().mean()))
res["rand_moy"] = statistics.mean(rands)
with log.open("a", encoding="utf-8") as f:
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        f.write(f"{k:20s} perte={v:.4f} (p={math.exp(v):8.3f})\n")
    f.write(f"\nsketch r=2 vs maxip : {res['sketch_r2'] - res['maxip_toutes_cles']:+.4f} nat "
            f"(octets de cles : 2/32 = 6,25 %)\n")
print("ok")
