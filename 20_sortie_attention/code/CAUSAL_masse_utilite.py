# -*- coding: utf-8 -*-
"""1) AUTO-TEST D'INVARIANCE CAUSALE : les pertes par position doivent etre identiques
   sur un prefixe (aucune dependance au futur). 2) RE-MESURE du test masse<->utilite
   avec le masque causal corrige. SmolLM2-135M, T=512, W=128, Lb=32 (12 blocs)."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W = 512, 32, 128
CFG = {"mode": "dense", "block": -1, "acc": []}

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
log = OUT / "CAUSAL_masse_utilite.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} | masque causal corrige\n", encoding="utf-8")


def poslosses(ids):
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
            if CFG["mode"] == "mass":
                S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                S = S.masked_fill((~causal)[None], float("-inf"))
                A = torch.softmax(S, dim=2)
                bm = (A * (~(j[None, :] >= (p - W + 1).clamp(min=0)[:, None]))[None].float()).view(H, qn, NB, Lb).sum(3)
                CFG["acc"].append(bm[:, W:, :].sum(dim=(0, 1)).detach())
                o = torch.einsum("hqk,hkd->hqd", A, ve[0])
            else:
                if CFG["block"] >= 0:
                    b0 = CFG["block"] * Lb
                    allowed |= ((j >= b0) & (j < b0 + Lb) & causal)[None].expand(H, qn, qn)
                o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = patched().__get__(layer.self_attn, type(layer.self_attn))

# --- 1) AUTO-TEST D'INVARIANCE CAUSALE (en mode m0, celui ou le bug etait) ---
CFG.update({"mode": "m0", "block": -1})
Lfull = poslosses(IDS)
Lpref = poslosses(IDS[:, :256])
delta = float((Lfull[:255] - Lpref).abs().max())
with log.open("a", encoding="utf-8") as f:
    f.write(f"AUTO-TEST CAUSAL (m0, prefixe 256) : ecart max par position = {delta:.3e} "
            f"-> {'OK' if delta < 1e-4 else 'ECHEC'}\n")
    f.flush()

# --- 2) TEST MASSE <-> UTILITE (corrige) ---
CFG.update({"mode": "mass", "acc": []})
_ = poslosses(IDS)
mass = torch.stack(CFG["acc"]).sum(0).numpy()
mass = mass / mass.sum()
CFG["mode"] = "m0"
l_m0 = float(poslosses(IDS).mean())
rows = []
for b in range(NB):
    CFG["block"] = b
    rows.append((b, float(mass[b]), float(poslosses(IDS).mean()) - l_m0))
CFG["block"] = -1
with log.open("a", encoding="utf-8") as f:
    f.write(f"m0_fenetre perte={l_m0:.4f}\nbloc | masse_norm | gain_de_perte (neg=utile)\n")
    for b, mm, dl in rows:
        f.write(f"{b:4d} | {mm:10.5f} | {dl:+.5f}\n")
    xs = [r[1] for r in rows]; ys = [r[2] for r in rows]
    mx = statistics.mean(xs); my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    corr = num / den if den > 0 else float("nan")
    f.write(f"\nCORRELATION masse<->gain = {corr:+.4f} (n={len(xs)})\n")
    f.write(f"blocs utiles (gain<0) : {sum(1 for y in ys if y < 0)}/{len(ys)} | "
            f"gain median={statistics.median(ys):+.5f} | meilleur={min(ys):+.5f} (bloc {rows[ys.index(min(ys))][0]}) | "
            f"bloc de masse max = bloc {rows[xs.index(max(xs))][0]} (gain {ys[xs.index(max(xs))]:+.5f})\n")
print("ok")
