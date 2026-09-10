# -*- coding: utf-8 -*-
"""GENERALITE sur une 3e architecture : GPT-2 (117M) - positions absolues apprises,
pas de RoPE, projection QKV combinee. Teste (a) la granularite a budget constant,
(b) la frontiere a 25 %, (c) le controle fenetre pure."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "gpt2"
T, DP, NFIT = 512, 8, 256
CONFIGS = [(128, 64, 1), (128, 32, 2), (128, 4, 16), (64, 4, 16), (128, 4, 0)]
OFFSETS = [0, 200000]
CFG = {"mode": "sparse", "sel": "pca", "W": 128, "Lb": 4, "m": 16}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
log = OUT / "gpt2_generalite.txt"
log.write_text(f"{MID} | T={T} d'={DP} | index PCA CAUSAL (pas de RoPE)\n"
               "W    Lb  m   cles  frac   offset  dense   maxip   pca     vs_dense  index\n",
               encoding="utf-8")


def make_fwd():
    def fwd(self, hidden_states=None, *args, **kwargs):
        if hidden_states is None:
            hidden_states = args[0]
        bsz, qn, _ = hidden_states.shape
        emb = self.embed_dim
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(emb, dim=2)
        sh = (bsz, qn, self.num_heads, self.head_dim)
        q = q.view(sh).transpose(1, 2)
        k = k.view(sh).transpose(1, 2)
        v = v.view(sh).transpose(1, 2)
        j = torch.arange(qn); p = torch.arange(qn)
        causal = (j[None, :] <= p[:, None])
        if CFG["mode"] == "dense":
            mask = causal[None, None].expand(bsz, self.num_heads, qn, qn).contiguous()
        else:
            W, Lb, M = CFG["W"], CFG["Lb"], CFG["m"]
            nb = qn // Lb
            mask = ((j[None, :] >= (p - W + 1).clamp(min=0)[:, None]) & causal)[None, None].expand(
                bsz, self.num_heads, qn, qn).clone()
            if M > 0:
                bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
                if CFG["sel"] == "maxip":
                    Sp = torch.einsum("bqd,bkd->bqk", q[0], k[0]) / math.sqrt(self.head_dim)
                else:
                    K = k[0]
                    C = torch.einsum("hnd,hne->hde", K[:, :NFIT, :], K[:, :NFIT, :])
                    _, ev = torch.linalg.eigh(C)
                    P = ev[:, :, -DP:].transpose(1, 2)
                    qp = torch.einsum("hqd,hpd->hqp", q[0], P)
                    kp = torch.einsum("hkd,hpd->hkp", K, P)
                    Sp = torch.einsum("hqp,hkp->hqk", qp, kp)
                Sb = Sp[:, :, :nb * Lb].view(self.num_heads, qn, nb, Lb)
                causb = causal[:, :nb * Lb].view(qn, nb, Lb)
                Sb = Sb.masked_fill((~causb)[None], float("-inf"))
                sc = Sb.max(dim=3).values.masked_fill((~bv).T[None], float("-inf"))
                tv, ti = torch.topk(sc, M, dim=2)
                ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
                valid = ti >= 0
                onehot = torch.zeros(self.num_heads, qn, nb, dtype=torch.bool)
                hh, pp, _ = torch.nonzero(valid, as_tuple=True)
                onehot[hh, pp, ti[valid]] = True
                mask |= (onehot[:, :, (j // Lb)] & causal[None])[None].expand(
                    bsz, self.num_heads, qn, qn)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(bsz, qn, emb)
        return self.c_proj(o), None
    return fwd


for layer in model.transformer.h:
    layer.attn.forward = make_fwd().__get__(layer.attn, type(layer.attn))


def loss(ids):
    with torch.no_grad():
        out = model(ids)
        return float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:]))


acc = {}
for off in OFFSETS:
    ids = ALL[off:off + T][None]
    CFG.update({"mode": "dense"}); ld = loss(ids)
    for (W, Lb, M) in CONFIGS:
        CFG.update({"mode": "sparse", "W": W, "Lb": Lb, "m": M})
        if M == 0:
            lp = lm = loss(ids)
        else:
            CFG.update({"sel": "maxip"}); lm = loss(ids)
            CFG.update({"sel": "pca"}); lp = loss(ids)
        acc.setdefault((W, Lb, M), []).append((ld, lm, lp))
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{W:4d} {Lb:4d} {M:3d} {W+M*Lb:5d} {(W+M*Lb)/T:6.3f} {off:7d} {ld:7.4f} "
                    f"{lm:7.4f} {lp:7.4f} {lm-ld:+9.4f} {lp-lm:+8.4f}\n")
            f.flush()
        print(f"off={off} W={W} Lb={Lb} m={M} vs_dense={lm-ld:+.4f} index={lp-lm:+.4f}", flush=True)
with log.open("a", encoding="utf-8") as f:
    f.write("\n=== moyennes 2 tranches ===\nW    Lb  m   cles  frac   vs_dense  index\n")
    for (W, Lb, M), v in acc.items():
        vd = sum(x[1] - x[0] for x in v) / len(v)
        ic = sum(x[2] - x[1] for x in v) / len(v)
        f.write(f"{W:4d} {Lb:4d} {M:3d} {W+M*Lb:5d} {(W+M*Lb)/T:6.3f} {vd:+9.4f} {ic:+8.4f}\n")
print("FIN")
