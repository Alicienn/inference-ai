# -*- coding: utf-8 -*-
"""kappa(T) : le cout par unite de masse piegee depend-il de la longueur de contexte ?
Masse piegee ~ constante en T (rho stable), mais Delta chute avec T dans la config
mise a l'echelle => kappa devrait chuter. Mesure directe : W=128, Lb in {4,16},
selection oracle top-m, m=64/Lb, a T=1024 et T=2048."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
W = 128
PMIN = 256
CFG = {"Lb": None}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]


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
        ve = v.repeat_interleave(g, 1)
        h = q.shape[1]
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2)
        del S
        Lb = CFG["Lb"]
        if Lb:
            m = 64 // Lb
            nb = qn // Lb
            B = A[:, :, :nb * Lb].view(h, qn, nb, Lb).sum(-1)      # masse par bloc
            bid = torch.arange(nb)
            elig = (bid[None, :] * Lb + Lb - 1) <= (torch.arange(qn) - W)[:, None]
            Bm = B * elig[None]
            _, sel = torch.topk(Bm, m, dim=2)                       # (h, qn, m)
            keys = (sel[..., None] * Lb + torch.arange(Lb)).reshape(h, qn, m * Lb)
            mask = torch.zeros(h, qn, qn, dtype=torch.bool)
            mask.scatter_(2, keys, True)
            mask |= (j[None, None, :] > (j[None, :, None] - W))
            A = A * mask
            A = A / A.sum(-1, keepdim=True).clamp(min=1e-9)
        o = torch.einsum("hqk,hkd->hqd", A, ve[0])
        del A
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))


def run(T, Lb, nchunks):
    CFG["Lb"] = Lb
    tot, n = 0.0, 0
    for c in range(nchunks):
        off = c * T
        ids = ALL[off:off + T][None]
        with torch.no_grad():
            logits = model(ids).logits[0]
        lg = logits[:-1]; tg = ids[0, 1:]
        ls = F.cross_entropy(lg, tg, reduction="none")
        tot += float(ls[PMIN:].sum()); n += int(ls[PMIN:].numel())
    return tot / n


log = OUT / "kappa_T.txt"
log.write_text(f"{MID} | W={W} | selection oracle top-m, m=64/Lb | p>={PMIN}\n"
               "T      Lb    perte_moyenne\n", encoding="utf-8")
res = {}
for T, nch in ((1024, 2), (2048, 1)):
    d = run(T, None, nch)
    l4 = run(T, 4, nch)
    l16 = run(T, 16, nch)
    res[T] = (d, l4, l16)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{T:5d}  dense {d:10.5f}\n{T:5d}      4 {l4:10.5f}  (Delta {l4-d:+.5f})\n"
                f"{T:5d}     16 {l16:10.5f}  (Delta {l16-d:+.5f})\n")
    print(f"T={T} dense={d:.5f} Lb4={l4:.5f} (D={l4-d:+.5f}) Lb16={l16:.5f} (D={l16-d:+.5f})",
          flush=True)
# masses piegees mesurees : rho(0) ~ 5,4e-4 (T=512/1024/2048) => M(4)=1,5*rho, M(16)=7,5*rho
rho = 5.4e-4
M4, M16 = 1.5 * rho, 7.5 * rho
for T, (d, l4, l16) in res.items():
    k = ((l16 - d) - (l4 - d)) / (M16 - M4)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"T={T} : kappa = ((D16)-(D4))/(M16-M4) = {k:.2f} nat/masse\n")
    print(f"T={T} kappa={k:.2f}", flush=True)
print("FIN")

