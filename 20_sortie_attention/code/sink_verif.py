# -*- coding: utf-8 -*-
"""VERIFICATION DU SINK + UNIVERSALITE DE LA DECOMPOSITION.
Question 1 : l'argmax de l'attention hors fenetre est-il vraiment au DEBUT de la
sequence (sink), ou juste "quelque part dans le premier decile" ?
Question 2 : la structure sink + fond plat tient-elle sur les 3 architectures ?"""
import sys, math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
T, W = 512, 128
OFFSETS = [0, 200000]
MID = {"smollm": "HuggingFaceTB/SmolLM2-135M", "qwen": "Qwen/Qwen2.5-0.5B",
       "gpt2": "gpt2"}[sys.argv[1]]

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
IS_GPT2 = hasattr(model, "transformer")
ACC = {"n": 0, "out": 0.0, "eff": 0.0, "top1": 0.0, "ampos": 0.0,
       "first10": 0, "first32": 0, "prof": torch.zeros(10), "amdec": torch.zeros(10)}


def stats(A, qn):
    pmin = W + 64
    if pmin >= qn:
        return
    for p in range(pmin, qn):
        oo = A[p, :p - W + 1]
        s = float(oo.sum())
        ACC["out"] += s
        if s > 1e-9:
            cond = oo / s
            ent = float(-(cond * (cond + 1e-12).log()).sum())
            ACC["eff"] += math.exp(ent)
            am = int(torch.argmax(cond))
            L = p - W + 1
            ACC["top1"] += float(cond[am])
            ACC["ampos"] += am / max(L - 1, 1)
            if am < 10:
                ACC["first10"] += 1
            if am < 32:
                ACC["first32"] += 1
            ACC["amdec"][min(9, int(10 * am / L))] += 1
            for b in range(10):
                a0 = int(b * L / 10); a1 = int((b + 1) * L / 10)
                ACC["prof"][b] += float(cond[a0:max(a1, a0 + 1)].sum())
        ACC["n"] += 1


def fwd_llama(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
    bsz, qn, _ = hidden_states.shape
    sh = (bsz, qn, -1, self.head_dim)
    q = self.q_proj(hidden_states).view(sh).transpose(1, 2)
    k = self.k_proj(hidden_states).view(sh).transpose(1, 2)
    v = self.v_proj(hidden_states).view(sh).transpose(1, 2)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
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
    stats(A, qn)
    del A
    return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None


def fwd_gpt2(self, hidden_states=None, *a, **kw):
    if hidden_states is None:
        hidden_states = a[0]
    bsz, qn, _ = hidden_states.shape
    emb = self.embed_dim
    q, k, v = self.c_attn(hidden_states).split(emb, dim=2)
    sh = (bsz, qn, self.num_heads, self.head_dim)
    q = q.view(sh).transpose(1, 2); k = k.view(sh).transpose(1, 2); v = v.view(sh).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    S = torch.einsum("hqd,hkd->hqk", q[0], k[0]) / math.sqrt(self.head_dim)
    j = torch.arange(qn)
    S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
    A = torch.softmax(S, dim=2).mean(dim=0)
    del S
    stats(A, qn)
    del A
    return self.c_proj(o.transpose(1, 2).reshape(bsz, qn, emb)), None


if IS_GPT2:
    for layer in model.transformer.h:
        layer.attn.forward = fwd_gpt2.__get__(layer.attn, type(layer.attn))
else:
    for layer in model.model.layers:
        layer.self_attn.forward = fwd_llama.__get__(layer.self_attn, type(layer.self_attn))

for off in OFFSETS:
    with torch.no_grad():
        model(ALL[off:off + T][None])

n = ACC["n"]
prof = ACC["prof"] / n
amdec = ACC["amdec"] / max(int(ACC["amdec"].sum()), 1)
log = OUT / f"sink_verif_{sys.argv[1]}.txt"
log.write_text(f"{MID} | T={T} W={W} | requetes={n}\n"
               f"masse hors fenetre        = {ACC['out']/n:.4f}\n"
               f"support effectif          = {ACC['eff']/n:.2f}\n"
               f"masse top-1 (moyenne)     = {ACC['top1']/n:.4f}\n"
               f"position normalisee argmax= {ACC['ampos']/n:.4f} (0 = debut de sequence)\n"
               f"argmax dans les 10 1res   = {ACC['first10']/n:.4f}\n"
               f"argmax dans les 32 1res   = {ACC['first32']/n:.4f}\n\n"
               "decile de POSITION de l'argmax :\n", encoding="utf-8")
for b in range(10):
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{b+1:5d} {float(amdec[b]):12.4f}\n")
with log.open("a", encoding="utf-8") as f:
    f.write("\nprofil conditionnel par decile de distance :\n")
    for b in range(10):
        f.write(f"{b+1:5d} {float(prof[b]):12.4f}\n")
print(f"[{sys.argv[1]}] out={ACC['out']/n:.4f} eff={ACC['eff']/n:.2f} top1={ACC['top1']/n:.4f} "
      f"ampos={ACC['ampos']/n:.4f} first10={ACC['first10']/n:.4f} first32={ACC['first32']/n:.4f}")
print("decile argmax:", [round(float(amdec[b]), 4) for b in range(10)])
print("profil:", [round(float(prof[b]), 4) for b in range(10)])
print("FIN")
