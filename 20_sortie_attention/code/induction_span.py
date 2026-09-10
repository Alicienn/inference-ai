# -*- coding: utf-8 -*-
"""SONDE D'INDUCTION PROPRE : span naturel de 64 tokens repete 4 fois.
Diagnostic prealable : la masse dense sur la COPIE 1 vue depuis la COPIE 4 doit
depasser le niveau uniforme (64/256=0,25) ; sinon le modele n'utilise pas la copie
distante et la retention creuse ne mesure rien."""
import math, pathlib, torch, torch.nn.functional as F, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, NBASE, DP, LB, NT, SPAN = 256, 256, 8, 4, 4, 64
QPOS = list(range(193, 256))
CFG = {"mode": "dense", "W": 32, "m": 16}
BASE = {}
ACC = {"md": 0.0, "ms": 0.0, "sel": 0, "n": 0, "nl": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]


def make_fwd(li):
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
        if CFG["mode"] == "collect":
            K = ke[0].float(); Kc = K - K.mean(1, keepdim=True)
            U, _, _ = torch.linalg.svd(Kc.transpose(1, 2), full_matrices=False)
            BASE[li] = U[:, :, :DP].contiguous()
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2)
        del S
        W, m = CFG["W"], CFG["m"]
        if CFG["mode"] == "pca" and m > 0 and qn == T:
            nb = qn // LB
            U = BASE[li]
            qp = torch.einsum("hqd,hde->hqe", q[0], U)
            kp = torch.einsum("hkd,hde->hke", ke[0], U)
            sc = (qp[:, :, None, :] * kp[:, None, :, :]).amax(-1)
            B = sc[:, :, :nb * LB].view(h, qn, nb, LB).amax(-1)
            del qp, kp, sc
            bid = torch.arange(nb)
            elig = (bid[None, :] * LB + LB - 1) <= (torch.arange(qn) - W)[:, None]
            _, sel = torch.topk(B * elig[None], min(m, nb), dim=2)
            keys = (sel[..., None] * LB + torch.arange(LB)).reshape(h, qn, -1)
            mask = torch.zeros(h, qn, qn, dtype=torch.bool)
            mask.scatter_(2, keys, True)
            mask |= (j[None, None, :] > (j[None, :, None] - W))
            md = A[:, QPOS, 0:SPAN].sum(-1)
            sp = A * mask
            ms = sp[:, QPOS, 0:SPAN].sum(-1)
            ACC["md"] += float(md.mean()); ACC["ms"] += float(ms.mean())
            ACC["sel"] += int(mask[:, QPOS, 0:SPAN].any(-1).sum())
            ACC["n"] += mask[:, QPOS, 0:SPAN].any(-1).numel(); ACC["nl"] += 1
            A = sp / sp.sum(-1, keepdim=True).clamp(min=1e-9)
            del mask, B, sp
        o = torch.einsum("hqk,hkd->hqd", A, ve[0])
        del A
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))

CFG["mode"] = "collect"
with torch.no_grad():
    model(ALL[:NBASE][None])
print("base collectee", flush=True)

seqs = []
rng = random.Random(11)
for i in range(NT):
    s = rng.randint(2000, len(ALL) - SPAN - 8)
    span = ALL[s:s + SPAN].clone()
    seqs.append(span.repeat(4))

log = OUT / "induction_span.txt"
log.write_text(f"{MID} | span naturel {SPAN} tok repete 4x (T={T}) | index PCA d'={DP} max\n"
               "config       perte_copie1  perte_copie4  gain_induction\n", encoding="utf-8")


def run(frac):
    if frac is None:
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
    else:
        CFG["mode"] = "pca"
        CFG["W"] = max(8, int(T * frac / 3)); CFG["m"] = max(1, int((T * frac * 2 / 3) // LB))
        for k in ACC: ACC[k] = 0
    l1 = l4 = 0.0
    for seq in seqs:
        with torch.no_grad():
            logits = model(seq[None]).logits[0]
        ls = F.cross_entropy(logits[:-1], seq[1:], reduction="none")
        l1 += float(ls[1:SPAN].mean()); l4 += float(ls[193:T - 1].mean())
    l1 /= NT; l4 /= NT
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{'dense' if frac is None else f'pca {frac:.3f}':12s} {l1:12.5f}  "
                f"{l4:12.5f}  {l1 - l4:+8.5f}\n")
    msg = f"{'dense' if frac is None else f'pca {frac:.3f}'} c1={l1:.5f} c4={l4:.5f} gain={l1-l4:+.5f}"
    if frac is not None:
        md = ACC["md"] / ACC["nl"]; ms = ACC["ms"] / ACC["nl"]
        msg += (f" | masse_dense={md:.4f} (uniforme {SPAN/T:.3f}) retenue={ms:.4f} "
                f"bloc_sel={100*ACC['sel']/max(ACC['n'],1):.1f}%")
    print(msg, flush=True)


run(None)
run(0.375)
run(0.125)
print("FIN")
