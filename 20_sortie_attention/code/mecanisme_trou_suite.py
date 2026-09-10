# -*- coding: utf-8 -*-
"""Suite du test du mecanisme : (a) profil de decroissance DERRIERE la fenetre,
(b) regression lineaire perte ~ kappa * masse_du_trou."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, W = 512, 128
LBS = [1, 2, 4, 8, 16, 32, 64]
OFFSETS = [0, 200000]
MESURE = {4: 0.0208, 8: 0.0594, 16: 0.1145, 32: 0.2145, 64: 0.3848}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
STORE = []


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
        STORE.append(torch.softmax(S, dim=2).mean(dim=0))
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))

prof_out = torch.zeros(60)
prof_in = torch.zeros(60)
n = 0
acc_mass = {Lb: [] for Lb in LBS}
for off in OFFSETS:
    STORE.clear()
    with torch.no_grad():
        model(ALL[off:off + T][None])
    A = torch.stack(STORE).mean(dim=0)
    STORE.clear()
    for p in range(W + 60, T):
        qlen = p - W + 1
        for d in range(1, 61):
            prof_out[d - 1] += A[p, qlen - d]
            prof_in[d - 1] += A[p, p - d + 1]
        n += 1
        for Lb in LBS:
            start = (qlen // Lb) * Lb
            acc_mass[Lb].append(float(A[p, start:qlen].sum()) if start <= qlen - 1 else 0.0)
prof_out /= n; prof_in /= n

log = OUT / "mecanisme_trou_suite.txt"
log.write_text(f"{MID} | T={T} W={W} | profil moyenne sur tetes, couches, requetes\n"
               "distance  dans_fenetre  derriere_fenetre\n", encoding="utf-8")
for d in range(1, 61):
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{d:8d} {float(prof_in[d-1]):14.8f} {float(prof_out[d-1]):16.8f}\n")

xs = [sum(acc_mass[Lb]) / len(acc_mass[Lb]) for Lb in MESURE]
ys = [MESURE[Lb] for Lb in MESURE]
mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
sxx = sum((x - mx) ** 2 for x in xs)
syy = sum((y - my) ** 2 for y in ys)
pente = sxy / sxx
ordonnee = my - pente * mx
r2 = sxy ** 2 / (sxx * syy)
with log.open("a", encoding="utf-8") as f:
    f.write("\n=== regression perte = kappa * masse_du_trou + c ===\n")
    f.write(f"kappa = {pente:.3f} nat par unite de masse\n")
    f.write(f"c     = {ordonnee:+.5f} nat\n")
    f.write(f"R2    = {r2:.6f}\n")
    f.write(f"points : {[round(x, 5) for x in xs]} -> {ys}\n")
print(f"kappa={pente:.3f} c={ordonnee:+.5f} R2={r2:.6f}")
print("profil derriere (d=1..10):", [round(float(prof_out[d]), 6) for d in range(10)])
print("FIN")
