# -*- coding: utf-8 -*-
"""AUDIT CAUSAL DE LA BASE : la base PCA du harnais est estimee sur les T cles du
segment, donc sur des cles FUTURES pour la requete p. Variante causale = base
estimee sur les 256 premieres cles seulement. Test APPARIE sur les 3 tranches."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "Qwen/Qwen2.5-0.5B"
T, Lb, W, M, NFIT = 512, 32, 128, 2, 256
OFFSETS = [0]
CFG = {"mode": "dense", "sel": "maxip", "dp": 8, "frozen": False, "check": True}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
log = OUT / "audit_causal_qwen.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} NFIT={NFIT}\n"
               "AUDIT : base PCA sur tout le segment (transductif) vs base causale\n", encoding="utf-8")


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
            ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
            nb = qn // Lb
            j = torch.arange(qn); p = torch.arange(qn)
            causal = (j[None, :] <= p[:, None])
            allowed = ((j[None, :] >= (p - W + 1).clamp(min=0)[:, None]) & causal)[None].expand(H, qn, qn).clone()
            bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
            sel = CFG["sel"]
            if sel == "rand":
                gen = torch.Generator().manual_seed(1000)
                ti = torch.randint(0, nb, (H, qn, M), generator=gen)
                ok = bv.T[None].expand(H, qn, nb).gather(2, ti.clamp(max=nb - 1))
                ti = torch.where(ok, ti, torch.full_like(ti, -1))
            else:
                if sel == "maxip":
                    Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                else:
                    dp = CFG["dp"]; K = ke[0]
                    Kfit = K[:, :NFIT, :] if CFG["frozen"] else K
                    C = torch.einsum("hnd,hne->hde", Kfit, Kfit)
                    _, ev = torch.linalg.eigh(C)
                    P = ev[:, :, -dp:].transpose(1, 2)
                    qp = torch.einsum("hqd,hpd->hqp", q[0], P)
                    kp = torch.einsum("hkd,hpd->hkp", K, P)
                    Sp = torch.einsum("hqp,hkp->hqk", qp, kp)
                Sb = Sp[:, :, :nb * Lb].view(H, qn, nb, Lb)
                causb = causal[:, :nb * Lb].view(qn, nb, Lb)
                Sb = Sb.masked_fill((~causb)[None], float("-inf"))
                sc = Sb.max(dim=3).values.masked_fill((~bv).T[None], float("-inf"))
                tv, ti = torch.topk(sc, M, dim=2)
                ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
            valid = ti >= 0
            onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
            hh, pp, _ = torch.nonzero(valid, as_tuple=True)
            onehot[hh, pp, ti[valid]] = True
            allowed |= onehot[:, :, (j // Lb)] & causal[None]
            if CFG["check"] and qn == T:
                with log.open("a", encoding="utf-8") as f:
                    f.write(f"CONTROLE : p=100 -> {int(allowed[0,100].sum())} ; "
                            f"p=300 -> {int(allowed[0,300].sum())}\n")
                    f.flush()
                CFG["check"] = False
            o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))


def loss(ids):
    with torch.no_grad():
        out = model(ids)
        return float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:]))


rows = {}
for off in OFFSETS:
    ids = ALL[off:off + T][None]
    r = {}
    CFG.update({"mode": "sparse", "sel": "maxip"}); r["maxip"] = loss(ids)
    CFG.update({"sel": "pca", "dp": 8, "frozen": False}); r["pca_d8_transductif"] = loss(ids)
    CFG.update({"sel": "pca", "dp": 8, "frozen": True}); r["pca_d8_causal"] = loss(ids)
    CFG.update({"sel": "pca", "dp": 16, "frozen": True}); r["pca_d16_causal"] = loss(ids)
    rows[off] = r
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\noffset {off}\n")
        for kk, vv in r.items():
            f.write(f"  {kk:20s} {vv:.4f}  (ecart {vv - r['maxip']:+.4f})\n")
        f.flush()

import statistics
with log.open("a", encoding="utf-8") as f:
    f.write("\n=== moyennes sur 3 tranches ===\n")
    keys = ["maxip", "pca_d8_transductif", "pca_d8_causal", "pca_d16_causal"]
    moy = {kk: statistics.mean([rows[o][kk] for o in OFFSETS]) for kk in keys}
    for kk in keys:
        f.write(f"{kk:20s} moy={moy[kk]:.4f}  ecart={moy[kk] - moy['maxip']:+.4f}\n")
    f.write(f"\nEFFET DE LA FUITE FUTURE (transductif - causal) : "
            f"{moy['pca_d8_transductif'] - moy['pca_d8_causal']:+.4f} nat\n")
print("ok")
