# -*- coding: utf-8 -*-
"""DECOMPOSITION LOCALE + DISTANTE. Question : la masse d'attention HORS fenetre
est-elle etalee (selection intrinsequement dure) ou concentree sur quelques cles
(selection facile, ce que suppose ASP) ? On mesure, sur la distribution conditionnelle
hors fenetre : le profil par distance (forme du creux), le support effectif
(exp de l'entropie) et la masse des top-k cles."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, W = 512, 128
OFFSETS = [0, 200000]
CFG = {"W": W}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
ACC = {"out_mass": 0.0, "eff": 0.0, "top": [0.0] * 5, "n": 0, "prof": torch.zeros(10)}


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
        Wl = CFG["W"]
        pmin = Wl + 64
        if pmin < qn:
            for p in range(pmin, qn):
                oo = A[p, :p - Wl + 1]
                s = float(oo.sum())
                ACC["out_mass"] += s
                if s > 1e-9:
                    cond = oo / s
                    ent = float(-(cond * (cond + 1e-12).log()).sum())
                    ACC["eff"] += math.exp(ent)
                    sv, _ = torch.sort(cond, descending=True)
                    for i, kk in enumerate((1, 4, 16, 64, 256)):
                        ACC["top"][i] += float(sv[:kk].sum())
                    L = p - Wl + 1
                    for b in range(10):
                        a0 = int(b * L / 10); a1 = int((b + 1) * L / 10)
                        ACC["prof"][b] += float(cond[a0:max(a1, a0 + 1)].sum())
                ACC["n"] += 1
        del A
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))
for off in OFFSETS:
    with torch.no_grad():
        model(ALL[off:off + T][None])

n = ACC["n"]
prof = ACC["prof"] / n
log = OUT / "decomposition_distante.txt"
log.write_text(f"{MID} | T={T} W={W} | distribution conditionnelle HORS fenetre\n"
               f"requetes={n}\n\n"
               f"masse hors fenetre (moyenne)      = {ACC['out_mass']/n:.4f}\n"
               f"support effectif exp(H)           = {ACC['eff']/n:.2f} cles\n"
               f"masse top-1                       = {ACC['top'][0]/n:.4f}\n"
               f"masse top-4                       = {ACC['top'][1]/n:.4f}\n"
               f"masse top-16                      = {ACC['top'][2]/n:.4f}\n"
               f"masse top-64                      = {ACC['top'][3]/n:.4f}\n"
               f"masse top-256                     = {ACC['top'][4]/n:.4f}\n\n"
               "profil conditionnel par distance (1 = juste derriere la fenetre)\n"
               "decile  fraction de la masse hors fenetre\n", encoding="utf-8")
for b in range(10):
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{b+1:5d} {float(prof[b]):14.4f}\n")
print(f"out_mass={ACC['out_mass']/n:.4f} eff={ACC['eff']/n:.2f} "
      f"top1={ACC['top'][0]/n:.4f} top4={ACC['top'][1]/n:.4f} top16={ACC['top'][2]/n:.4f} "
      f"top64={ACC['top'][3]/n:.4f}")
print("profil:", [round(float(prof[b]), 4) for b in range(10)])
print("FIN")
