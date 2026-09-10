# -*- coding: utf-8 -*-
"""INVARIANCE DE kappa ET rho. Mesure la masse du trou et la densite plate derriere
la fenetre sur 3 modeles, puis teste la prediction HORS ECHANTILLON :
   Delta(Lb) - Delta(4)  ~=  kappa_SmolLM2 * (M(Lb) - M(4))
ou kappa_SmolLM2 = 11,682 vient de SmolLM2 et n'est PAS reajuste."""
import sys, math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
T, W = 512, 128
LBS = [4, 8, 16, 32, 64]
OFFSETS = [0, 200000]
KAPPA_SMOLLM = 11.682

SPECS = {
    "smollm": ("HuggingFaceTB/SmolLM2-135M", {4: 0.0208, 8: 0.0594, 16: 0.1145, 32: 0.2145, 64: 0.3848}),
    "qwen":   ("Qwen/Qwen2.5-0.5B",          {4: -0.0033, 8: 0.0350, 16: 0.1320, 32: 0.3460, 64: 0.8020}),
    "gpt2":   ("gpt2",                       {4: -0.0041, 8: None, 16: None, 32: 0.2158, 64: 0.3997}),
}
which = sys.argv[1]
MID, MESURE = SPECS[which]

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
STORE = []
IS_GPT2 = hasattr(model, "transformer")


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
    STORE.append(torch.softmax(S, dim=2).mean(dim=0))
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
    STORE.append(torch.softmax(S, dim=2).mean(dim=0))
    return self.c_proj(o.transpose(1, 2).reshape(bsz, qn, emb)), None


if IS_GPT2:
    for layer in model.transformer.h:
        layer.attn.forward = fwd_gpt2.__get__(layer.attn, type(layer.attn))
else:
    for layer in model.model.layers:
        layer.self_attn.forward = fwd_llama.__get__(layer.self_attn, type(layer.self_attn))

PMIN = W + max(LBS)
acc = {Lb: [] for Lb in LBS}
prof = torch.zeros(64)
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
            prof[d - 1] += A[p, qlen - d]
        n += 1
        for Lb in LBS:
            start = (qlen // Lb) * Lb
            acc[Lb].append(float(A[p, start:qlen].sum()) if start <= qlen - 1 else 0.0)
prof /= n

log = OUT / f"invariance_{which}.txt"
log.write_text(f"{MID} | T={T} W={W} | p>={PMIN}\nLb  masse_trou  perte  Delta-Delta4  predit\n",
               encoding="utf-8")
M = {Lb: sum(acc[Lb]) / len(acc[Lb]) for Lb in LBS}
base_m, base_d = M[4], MESURE[4]
lignes = []
for Lb in LBS:
    if MESURE[Lb] is None:
        continue
    obs = MESURE[Lb] - base_d
    pred = KAPPA_SMOLLM * (M[Lb] - base_m)
    lignes.append((Lb, M[Lb], MESURE[Lb], obs, pred))
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{Lb:3d} {M[Lb]:11.6f} {MESURE[Lb]:+8.4f} {obs:+12.4f} {pred:+10.4f}\n")
xs = [l[3] for l in lignes]; ys = [l[4] for l in lignes]
mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
sxy = sum((a-mx)*(b-my) for a, b in zip(xs, ys)); sxx = sum((a-mx)**2 for a in xs)
syy = sum((b-my)**2 for b in ys)
pente = sxy/sxx if sxx > 0 else float("nan")
r2 = (sxy**2/(sxx*syy)) if (sxx > 0 and syy > 0) else float("nan")
with log.open("a", encoding="utf-8") as f:
    f.write(f"\nrho (densite plate, d=4..60) = {float(prof[3:60].mean()):.3e} par cle\n")
    f.write(f"rho*(Lb-1)/2 predit la masse : ")
    for Lb in LBS:
        f.write(f"Lb={Lb} {float(prof[3:60].mean())*((Lb-1)/2):.6f} vs {M[Lb]:.6f} ; ")
    f.write(f"\npente obs/predit = {pente:.3f} ; R2 = {r2:.4f}\n")
print(f"[{which}] rho={float(prof[3:60].mean()):.3e} pente={pente:.3f} R2={r2:.4f}")
for l in lignes:
    print(f"  Lb={l[0]:3d} masse={l[1]:.5f} obs={l[3]:+.4f} predit={l[4]:+.4f}")
print("FIN")
