# -*- coding: utf-8 -*-
"""BALAYAGE (Lb, m) avec l'index PCA causal d'=8. T=512, W=128, tranche 0.
Mesure : ecart a maxip pour la MEME configuration + fraction de cles lues."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, W, DP, NFIT = 512, 128, 8, 256
CFG = {"mode": "sparse", "sel": "pca", "Lb": 32, "m": 2}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
log = OUT / "balayage_Lb_m_2.txt"
log.write_text(f"{MID} | T={T} W={W} d'={DP} NFIT={NFIT} | index PCA CAUSAL\n", encoding="utf-8")


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
        if CFG["mode"] == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(g > 1))
        else:
            Lb, M = CFG["Lb"], CFG["m"]
            ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
            nb = qn // Lb
            j = torch.arange(qn); p = torch.arange(qn)
            causal = (j[None, :] <= p[:, None])
            allowed = ((j[None, :] >= (p - W + 1).clamp(min=0)[:, None]) & causal)[None].expand(
                q.shape[1], qn, qn).clone()
            bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
            if CFG["sel"] == "maxip":
                Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
            else:
                K = ke[0]
                C = torch.einsum("hnd,hne->hde", K[:, :NFIT, :], K[:, :NFIT, :])
                _, ev = torch.linalg.eigh(C)
                P = ev[:, :, -DP:].transpose(1, 2)
                qp = torch.einsum("hqd,hpd->hqp", q[0], P)
                kp = torch.einsum("hkd,hpd->hkp", K, P)
                Sp = torch.einsum("hqp,hkp->hqk", qp, kp)
            Sb = Sp[:, :, :nb * Lb].view(q.shape[1], qn, nb, Lb)
            causb = causal[:, :nb * Lb].view(qn, nb, Lb)
            Sb = Sb.masked_fill((~causb)[None], float("-inf"))
            sc = Sb.max(dim=3).values.masked_fill((~bv).T[None], float("-inf"))
            tv, ti = torch.topk(sc, M, dim=2)
            ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
            valid = ti >= 0
            onehot = torch.zeros(q.shape[1], qn, nb, dtype=torch.bool)
            hh, pp, _ = torch.nonzero(valid, as_tuple=True)
            onehot[hh, pp, ti[valid]] = True
            allowed |= onehot[:, :, (j // Lb)] & causal[None]
            o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))

def loss(ids):
    with torch.no_grad():
        out = model(ids)
        return float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:]))


for off in [200000, 400000]:
    ids = ALL[off:off + T][None]
    with log.open("a", encoding="utf-8") as f:
        CFG.update({"mode": "dense"})
        f.write(f"\n=== offset {off} === dense {loss(ids):.4f}\n")
        f.write("Lb   m   maxip    pca_d8   ecart     frac_cles_lues\n")
        f.flush()
    for Lb in [16, 32, 64]:
        for M in [1, 2, 4]:
            CFG.update({"mode": "sparse", "sel": "maxip", "Lb": Lb, "m": M})
            lm = loss(ids)
            CFG.update({"sel": "pca"})
            lp = loss(ids)
            frac = (M * Lb + W) / T
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{Lb:3d} {M:3d} {lm:8.4f} {lp:8.4f} {lp - lm:+8.4f}  {frac:6.3f}\n")
                f.flush()
            print(f"off={off} Lb={Lb} m={M} ecart={lp-lm:+.4f}", flush=True)
print("FIN")
