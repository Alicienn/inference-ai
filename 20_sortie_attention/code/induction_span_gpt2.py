# -*- coding: utf-8 -*-
"""Troisieme architecture : GPT-2 (sans RoPE). Meme sonde que SmolLM2/Qwen :
span naturel de 64 tokens repete 4x, T=256. On capture l'attention NATIVE via
output_attentions (aucun patch du forward) et on mesure la masse sur la COPIE 1
(source distante, uniforme 0,250) et la COPIE 3 (source proche)."""
import pathlib, torch, torch.nn.functional as F, random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "gpt2"
T, NT, SPAN = 256, 4, 64
QPOS = list(range(193, 256))

tok = GPT2Tokenizer.from_pretrained(MID)
model = GPT2LMHeadModel.from_pretrained(MID, attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]

seqs = []
rng = random.Random(11)
for i in range(NT):
    s = rng.randint(2000, len(ALL) - SPAN - 8)
    seqs.append(ALL[s:s + SPAN].clone().repeat(4))

log = OUT / "induction_span_gpt2.txt"
log.write_text(f"{MID} | span naturel {SPAN} tok repete 4x (T={T}) | attention native\n"
               "config        perte_copie1  perte_copie4  gain_induction\n", encoding="utf-8")

l1 = l4 = 0.0
m1 = m3 = 0.0
m1_max = 0.0; m1_max_layer = -1; m1_max_head = -1
for seq in seqs:
    with torch.no_grad():
        out = model(seq[None], output_attentions=True)
    ls = F.cross_entropy(out.logits[0, :-1], seq[1:], reduction="none")
    l1 += float(ls[1:SPAN].mean()); l4 += float(ls[193:T - 1].mean())
    for li, A in enumerate(out.attentions):          # (1, h, q, k)
        a = A[0]                                      # (h, q, k)
        c1 = a[:, QPOS, 0:SPAN].sum(-1)               # (h, len(QPOS))
        c3 = a[:, QPOS, 128:192].sum(-1)
        m1 += float(c1.mean()); m3 += float(c3.mean())
        v, idx = float(c1.max()), int(c1.argmax())
        if v > m1_max:
            m1_max, m1_max_layer, m1_max_head = v, li, idx
l1 /= NT; l4 /= NT
NL = len(model.transformer.h); NH = model.config.n_head
m1 /= (NT * NL); m3 /= (NT * NL)
with log.open("a", encoding="utf-8") as f:
    f.write(f"dense         {l1:12.5f}  {l4:12.5f}  {l1 - l4:+8.5f}\n")
    f.write(f"masse moyenne copie1={m1:.4f} copie3={m3:.4f} (uniforme {SPAN/T:.3f})\n")
    f.write(f"max couche/tete copie1={m1_max:.4f} (couche {m1_max_layer}, tete {m1_max_head})\n")
print(f"dense c1={l1:.5f} c4={l4:.5f} gain={l1-l4:+.5f}", flush=True)
print(f"masse copie1={m1:.4f} copie3={m3:.4f} (uniforme {SPAN/T:.3f})", flush=True)
print(f"max couche/tete copie1={m1_max:.4f} (couche {m1_max_layer}, tete {m1_max_head})", flush=True)
print("FIN")
