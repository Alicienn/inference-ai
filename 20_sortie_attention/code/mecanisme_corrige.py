# -*- coding: utf-8 -*-
"""Correction du confondant : on restreint aux requetes p >= W+64 pour que le trou
ne puisse pas atteindre la position 0 (sink d'attention). Recalcul de la masse du
trou, du profil derriere la fenetre, et de la regression perte ~ kappa * masse."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, W = 512, 128
LBS = [2, 4, 8, 16, 32, 64]
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

PMIN = W + max(LBS)          # 192 : le trou ne peut plus atteindre la position 0
prof_out = torch.zeros(64); prof_in = torch.zeros(64)
acc = {Lb: [] for Lb in LBS}
n = 0
for off in OFFSETS:
    STORE.clear()
    with torch.no_grad():
        model(ALL[off:off + T][None])
    A = torch.stack(STORE).mean(dim=0)
    STORE.clear()
    for p in range(PMIN, T):
        qlen = p - W + 1
        for d in range(1, 65):
            prof_out[d - 1] += A[p, qlen - d]
            prof_in[d - 1] += A[p, p - d + 1]
        n += 1
        for Lb in LBS:
            start = (qlen // Lb) * Lb
            acc[Lb].append(float(A[p, start:qlen].sum()) if start <= qlen - 1 else 0.0)
prof_out /= n; prof_in /= n

log = OUT / "mecanisme_corrige.txt"
log.write_text(f"{MID} | T={T} W={W} | p >= {PMIN} (trou hors position 0)\n"
               "Lb   taille_moy  masse_trou   perte_mesuree  perte/masse\n", encoding="utf-8")
pts = []
for Lb in LBS:
    m = sum(acc[Lb]) / len(acc[Lb])
    s = sum((i % Lb) for i in range(1, T - W + 1)) / (T - W)
    mes = MESURE.get(Lb)
    if mes is not None:
        pts.append((m, mes))
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{Lb:3d} {s:11.3f} {m:12.6f} "
                f"{('' if mes is None else f'{mes:14.4f}'):>14s} "
                f"{('' if mes is None else f'{mes/m:10.2f}'):>10s}\n")
xs = [a for a, b in pts]; ys = [b for a, b in pts]
mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys)); sxx = sum((x-mx)**2 for x in xs)
syy = sum((y-my)**2 for y in ys)
pente = sxy/sxx; c0 = my - pente*mx; r2 = sxy**2/(sxx*syy)
with log.open("a", encoding="utf-8") as f:
    f.write(f"\nkappa = {pente:.3f} nat/masse ; c = {c0:+.5f} ; R2 = {r2:.6f}\n")
    f.write(f"points (masse, perte) : {[(round(a,5), b) for a, b in pts]}\n")
    f.write("\ndistance  dans_fenetre  derriere_fenetre\n")
    for d in range(1, 65):
        f.write(f"{d:8d} {float(prof_in[d-1]):14.8f} {float(prof_out[d-1]):16.8f}\n")
print(f"kappa={pente:.3f} c={c0:+.5f} R2={r2:.6f}")
print("masses:", [(Lb, round(sum(acc[Lb])/len(acc[Lb]), 5)) for Lb in LBS])
print("derriere d=1,8,16,32,64:", [round(float(prof_out[d-1]), 6) for d in (1,8,16,32,64)])
print("FIN")
