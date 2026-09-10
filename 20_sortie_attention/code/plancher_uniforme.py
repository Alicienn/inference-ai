# -*- coding: utf-8 -*-
"""HYPOTHESE DU PLANCHER UNIFORME : l'attention hors fenetre serait repartie
uniformement, d'ou rho = (1 - masse_fenetre)/(T - W). Test sur une grille (T, W).
On mesure : mu_in (masse dans la fenetre), rho_local (masse moyenne par cle juste
DERRIERE la fenetre, d=1..60), rho_global ((1-mu_in)/(nombre de cles hors fenetre))."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
GRID = [(512, 32), (512, 64), (512, 128), (512, 256),
        (1024, 128), (1024, 256), (2048, 256), (2048, 512)]
OFFSETS = [0]
CFG = {"W": 128}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
ACC = {}


def make_fwd():
    def fwd(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
        bsz, qn, _ = hidden_states.shape
        sh = (bsz, qn, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(sh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(sh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(sh).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        g = self.num_key_value_groups
        ke = k.repeat_interleave(g, 1)
        o = F.scaled_dot_product_attention(q, ke, v.repeat_interleave(g, 1), is_causal=True)
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2).mean(dim=0)
        del S
        W = CFG["W"]
        pmin = W + 64
        if pmin < qn:
            sub = A[pmin:qn]                                  # (n, qn)
            idx = torch.arange(pmin, qn)
            win = torch.stack([sub[i, max(0, p - W + 1):p + 1].sum() for i, p in enumerate(idx)])
            glob = torch.stack([(1.0 - win[i]) / (p - W + 1) for i, p in enumerate(idx)])
            loc = torch.stack([sub[i, max(0, p - W - 59):p - W + 1].mean()
                               for i, p in enumerate(idx)])
            key = (qn, W)
            acc = ACC.setdefault(key, [0.0, 0.0, 0.0, 0])
            acc[0] += float(win.mean()); acc[1] += float(glob.mean())
            acc[2] += float(loc.mean()); acc[3] += 1
        del A
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))

log = OUT / "plancher_uniforme.txt"
log.write_text(f"{MID} | test de rho = (1-mu_in)/(T-W)\n"
               "T     W    mu_in   rho_local  rho_global  ratio   rho*(T-W)\n", encoding="utf-8")
for (T, W) in GRID:
    CFG["W"] = W
    for off in OFFSETS:
        with torch.no_grad():
            model(ALL[off:off + T][None])
    k = (T, W)
    if k in ACC:
        n = ACC[k][3]
        mu = ACC[k][0] / n; rg = ACC[k][1] / n; rl = ACC[k][2] / n
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{T:5d} {W:4d} {mu:8.4f} {rl:10.6f} {rg:11.6f} {rl/rg:7.3f} "
                    f"{rl*(T-W):10.4f}\n")
        print(f"T={T} W={W} mu_in={mu:.4f} rho_local={rl:.6f} rho_global={rg:.6f} "
              f"ratio={rl/rg:.3f}", flush=True)
print("FIN")
