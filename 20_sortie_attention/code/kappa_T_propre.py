# -*- coding: utf-8 -*-
"""kappa(T) PROPRE : fraction lue constante (W=T/8, m*Lb=T/4 => 37,5 % des cles)
a T=1024 (2 tranches) et T=2048 (1 tranche, meme texte). La masse piegee est
MESUREE dans le meme passage (masse dans la clef de bloc partielle derriere la fenetre),
pas empruntee a rho. kappa(T) = (Delta16 - Delta4)/(M16 - M4)."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
PMIN = 320
CFG = {"Lb": None, "W": 128, "m": 0}
MASS = {4: {"n": 0, "s": 0.0}, 16: {"n": 0, "s": 0.0}}

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
        ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
        h = q.shape[1]
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2)
        del S
        Lb = CFG["Lb"]; W = CFG["W"]
        if Lb is None:
            Am = A.mean(0)
            for Lbv in (4, 16):
                pmin = max(PMIN, W + Lbv + 1)
                if pmin < qn - 1:
                    idx = torch.arange(pmin, qn)
                    bstar = ((idx - W + 1) // Lbv) - 1
                    start = (bstar + 1) * Lbv
                    pos = torch.arange(qn)[None, :]
                    gap = (pos >= start[:, None]) & (pos <= (idx - W)[:, None])
                    MASS[Lbv]["s"] += float((Am[pmin:qn] * gap).sum())
                    MASS[Lbv]["n"] += len(idx)
        else:
            m = CFG["m"]; nb = qn // Lb
            B = A[:, :, :nb * Lb].view(h, qn, nb, Lb).sum(-1)
            bid = torch.arange(nb)
            elig = (bid[None, :] * Lb + Lb - 1) <= (torch.arange(qn) - W)[:, None]
            _, sel = torch.topk(B * elig[None], m, dim=2)
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


def run(T, nchunks, Lb):
    W = T // 8
    CFG.update({"Lb": Lb, "W": W, "m": (T // 4) // Lb if Lb else 0})
    tot, n = 0.0, 0
    for c in range(nchunks):
        ids = ALL[c * T:(c + 1) * T][None]
        with torch.no_grad():
            logits = model(ids).logits[0]
        ls = F.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
        tot += float(ls[PMIN:].sum()); n += int(ls[PMIN:].numel())
    return tot / n


log = OUT / "kappa_T_propre.txt"
log.write_text(f"{MID} | fraction lue 37,5 % (W=T/8, m*Lb=T/4) | p>={PMIN}\n"
               "T      Lb   perte      Delta    masse_piegee\n", encoding="utf-8")
res = {}
for T, nch in ((1024, 2), (2048, 1)):
    d = run(T, nch, None)
    M4 = MASS[4]["s"] / max(MASS[4]["n"], 1)
    M16 = MASS[16]["s"] / max(MASS[16]["n"], 1)
    for kk in (4, 16):
        MASS[kk]["s"] = MASS[kk]["n"] = 0
    l4 = run(T, nch, 4)
    l16 = run(T, nch, 16)
    res[T] = (d, l4, l16, M4, M16)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{T:5d}  dense {d:10.5f}  ---        ---\n")
        f.write(f"{T:5d}      4 {l4:10.5f}  {l4-d:+.5f}  {M4:.3e}\n")
        f.write(f"{T:5d}     16 {l16:10.5f}  {l16-d:+.5f}  {M16:.3e}\n")
    print(f"T={T} dense={d:.5f} Lb4={l4:.5f}(D{l4-d:+.5f}, M={M4:.3e}) "
          f"Lb16={l16:.5f}(D{l16-d:+.5f}, M={M16:.3e})", flush=True)
for T, (d, l4, l16, M4, M16) in res.items():
    k = ((l16 - d) - (l4 - d)) / (M16 - M4) if M16 != M4 else float("nan")
    with log.open("a", encoding="utf-8") as f:
        f.write(f"T={T} : kappa = {k:.2f} nat/masse\n")
    print(f"T={T} kappa={k:.2f}", flush=True)
print("FIN")
