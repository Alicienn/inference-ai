# -*- coding: utf-8 -*-
"""REPLICATION du test masse<->utilite sur un SECOND modele (Qwen2.5-0.5B).
Protocole identique a masse_vs_utilite.py (SmolLM2) : T=512, W=128, Lb=32, 12 blocs candidats,
corpus local. Pour chaque bloc : masse d'attention exacte et gain de perte s'il est ajoute seul."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "Qwen/Qwen2.5-0.5B"
T, Lb, W = 512, 32, 128
CFG = {"mode": "dense", "block": -1, "acc": []}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
KVH = getattr(cfg, "num_key_value_heads", H)
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
mt = cfg.model_type
if mt == "qwen2":
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
elif mt == "llama":
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
else:
    raise SystemExit(f"modele non gere: {mt}")

corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ids = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0, :T][None]
NB = T // Lb
log = OUT / "masse_vs_utilite_qwen.txt"
log.write_text(f"{MID} | T={T} | W={W} | Lb={Lb} | blocs={NB} | H={H} KVH={KVH} D={D}\n",
               encoding="utf-8")


def loss():
    with torch.no_grad():
        out = model(ids)
        return float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none").mean())


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
            j = torch.arange(qn); p = torch.arange(qn)
            win = (j[None, :] >= (p - W + 1).clamp(min=0)[:, None])
            if CFG["mode"] == "mass":
                S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                caus = (j[None, :] <= p[:, None])
                S = S.masked_fill((~caus)[None], float("-inf"))
                A = torch.softmax(S, dim=2)
                bm = (A * (~win)[None].float()).view(H, qn, NB, Lb).sum(3)
                CFG["acc"].append(bm[:, W:, :].sum(dim=(0, 1)).detach())
                o = torch.einsum("hqk,hkd->hqd", A, ve[0])
            else:
                allowed = win[None].expand(H, qn, qn).clone()
                if CFG["block"] >= 0:
                    b0 = CFG["block"] * Lb
                    blk = ((j >= b0) & (j < b0 + Lb) & (j[None, :] <= p[:, None]))[None]
                    allowed |= blk.expand(H, qn, qn)
                o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = patched().__get__(layer.self_attn, type(layer.self_attn))

CFG["mode"] = "mass"; CFG["acc"] = []
_ = loss()
mass = torch.stack(CFG["acc"]).sum(0).numpy()
mass = mass / mass.sum()
CFG["mode"] = "m0"; CFG["block"] = -1
l_m0 = loss()
rows = []
for b in range(NB):
    CFG["block"] = b
    rows.append((b, float(mass[b]), loss() - l_m0))

with log.open("a", encoding="utf-8") as f:
    f.write(f"m0_fenetre perte={l_m0:.4f}\nbloc | masse_norm | gain_de_perte\n")
    for b, mm, dl in rows:
        f.write(f"{b:4d} | {mm:10.5f} | {dl:+.5f}\n")
    xs = [r[1] for r in rows]; ys = [r[2] for r in rows]
    n = len(xs); mx = statistics.mean(xs); my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    corr = num / den if den > 0 else float("nan")
    nz = [(x, y) for x, y in zip(xs, ys) if x > 0]
    f.write(f"\nCORRELATION masse<->gain = {corr:+.4f} (n={n}, blocs a masse non nulle={len(nz)})\n")
    f.write(f"blocs UTILES (gain<0) : {sum(1 for y in ys if y < 0)}/{n} | "
            f"gain median={statistics.median(ys):+.5f} | meilleur={min(ys):+.5f} (bloc {rows[ys.index(min(ys))][0]})\n")
    f.write(f"bloc de masse MAX : bloc {rows[xs.index(max(xs))][0]} (masse {max(xs):.5f}, "
            f"gain {ys[xs.index(max(xs))]:+.5f})\n")
print("ok")
