# -*- coding: utf-8 -*-
"""PROFIL FIN DERRIERE LA FENETRE. rho(d) = masse moyenne par cle a distance d
derriere la fenetre (d=0 = cle juste derriere). Question : le gradient de recence
est-il une queue exponentielle de longueur lambda, et lambda depend-il de T ?
Si lambda >> Lb/2 = 2, l'approximation "fond plat" de Delta ~ kappa*rho*(Lb-1)/2 tient."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
TS = [512, 1024, 2048]
W, D = 128, 128
CFG = {"T": 512}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
ACC = {}


def stats(A, qn):
    pmin = W + D + 1
    if pmin >= qn - 1:
        return
    idx = torch.arange(pmin, qn)
    IDX = (idx - W)[:, None] - torch.arange(D)[None, :]          # (n, D)
    vals = A[pmin:qn].gather(1, IDX)                              # d = 0..D-1
    a = ACC.setdefault(qn, {"n": 0, "s": torch.zeros(D), "out": 0.0})
    a["n"] += len(idx)
    a["s"] += vals.sum(0)
    a["out"] += float((A[pmin:qn] * (torch.arange(qn)[None, :] <= (idx - W)[:, None])).sum())


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
        stats(A, qn)
        del A
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))

log = OUT / "profil_fin.txt"
log.write_text(f"{MID} | W={W} | rho(d) = masse par cle a distance d derriere la fenetre\n"
               "d     rho(T=512)   rho(T=1024)  rho(T=2048)\n", encoding="utf-8")
for T in TS:
    with torch.no_grad():
        model(ALL[:T][None])
curves = {}
for qn, a in ACC.items():
    curves[qn] = (a["s"] / a["n"]).tolist()
with log.open("a", encoding="utf-8") as f:
    for d in range(D):
        row = f"{d:4d}"
        for T in TS:
            row += f"  {curves[T][d]:11.3e}"
        f.write(row + "\n")
for T in TS:
    c = curves[T]
    # ajustement exponentiel sur d=4..63 (eviter le point de contact et la queue bruitee)
    xs = list(range(4, 64))
    ys = [math.log(c[d]) for d in xs]
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys)); sxx = sum((x-mx)**2 for x in xs)
    pente = sxy/sxx; lam = -1.0/pente if pente < 0 else float("inf")
    r2 = 1.0
    yy = [my + pente*(x-mx) for x in xs]
    ss = sum((y-yh)**2 for y, yh in zip(ys, yy)); st = sum((y-my)**2 for y in ys)
    r2 = 1 - ss/st if st > 0 else 0.0
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\nT={T} : rho(0)={c[0]:.3e} rho(63)={c[63]:.3e} "
                f"rapport={c[0]/c[63]:.2f} lambda={lam:.1f} cles (R2={r2:.3f})\n")
    print(f"T={T} rho(0)={c[0]:.3e} rho(63)={c[63]:.3e} ratio={c[0]/c[63]:.2f} "
          f"lambda={lam:.1f} R2={r2:.3f}", flush=True)
print("FIN")
