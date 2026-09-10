# -*- coding: utf-8 -*-
"""TEST QUANTITATIF DU MECANISME D'EFFET DE BORD.
Le trou d'une grille Lb devant une fenetre W a une taille (p-W+1) mod Lb.
Prediction : la perte vs dense est proportionnelle a la MASSE D'ATTENTION
qui tombe dans ce trou. On mesure cette masse sur les vraies distributions
d'attention (modele dense, softmax exacte) et on la compare aux pertes mesurees."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, W = 512, 128
LBS = [1, 2, 4, 8, 16, 32, 64]
OFFSETS = [0, 200000]
# pertes mesurees au meme budget constant (m*Lb=64, W=128), SmolLM2, 3 tranches
MESURE = {64: 0.3848, 32: 0.2145, 16: 0.1145, 8: 0.0594, 4: 0.0208}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]

STORE = []
CFG = {"mode": "probe"}


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
        o = F.scaled_dot_product_attention(q, ke, v.repeat_interleave(g, 1),
                                           is_causal=True, enable_gqa=False)
        # distributions d'attention exactes (modele dense) pour le diagnostic
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2)
        STORE.append(A.mean(dim=0).to(torch.float32))   # moyenne sur les tetes
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))

log = OUT / "mecanisme_trou.txt"
log.write_text(f"{MID} | T={T} W={W} | masse d'attention dans le trou (moyenne "
               "sur tetes et couches, requetes p>=W)\n"
               "Lb   taille_moy  masse_trou   perte_mesuree  perte/masse\n", encoding="utf-8")

acc_mass = {Lb: [] for Lb in LBS}
acc_size = {Lb: [] for Lb in LBS}
prof = torch.zeros(W)
nprof = 0
for off in OFFSETS:
    ids = ALL[off:off + T][None]
    STORE.clear()
    with torch.no_grad():
        model(ids)
    # STORE : une matrice A moyennee sur tetes par couche, (T, T)
    A = torch.stack(STORE).mean(dim=0)          # moyenne sur les couches
    STORE.clear()
    for p in range(W, T):
        qlen = p - W + 1
        prof += A[p, max(0, p - W + 1):p + 1].flip(0)[:W]   # profil par distance
        nprof += 1
        for Lb in LBS:
            start = (qlen // Lb) * Lb
            if start <= qlen - 1:
                acc_mass[Lb].append(float(A[p, start:qlen].sum()))
            else:
                acc_mass[Lb].append(0.0)
            acc_size[Lb].append(qlen % Lb)
prof /= max(nprof, 1)

for Lb in LBS:
    m = sum(acc_mass[Lb]) / len(acc_mass[Lb])
    s = sum(acc_size[Lb]) / len(acc_size[Lb])
    mes = MESURE.get(Lb)
    ratio = (mes / m) if (mes is not None and m > 0) else float("nan")
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{Lb:3d} {s:11.3f} {m:12.6f} {('' if mes is None else f'{mes:14.4f}'):>14s} "
                f"{('' if mes is None else f'{ratio:10.1f}'):>10s}\n")
    print(f"Lb={Lb:3d} taille={s:6.2f} masse={m:.6f} perte={mes}", flush=True)
with log.open("a", encoding="utf-8") as f:
    f.write("\n=== profil d'attention par distance derriere la fenetre (W=128) ===\n")
    f.write("distance  masse_moyenne\n")
    for d in range(1, 41):
        f.write(f"{d:8d} {float(prof[d-1]):14.8f}\n")
print("FIN")
