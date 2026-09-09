# -*- coding: utf-8 -*-
"""Loi d'echelle sous harnais CORRIGE : T = 512 / 1024 / 2048, W=128, Lb=32, m=2."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
Lb, W, M = 32, 128, 2
CFG = {"mode": "sparse", "sel": "maxip", "dp": 8, "check": True}
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
log = OUT / "echelle_corrigee.txt"
log.write_text(f"{MID} | W={W} Lb={Lb} m={M} | HARNAIS CORRIGE\n", encoding="utf-8")


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
                    C = torch.einsum("hnd,hne->hde", K, K)
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
            if CFG["check"] and qn >= 1024:
                c = int(allowed[0, 1000].sum())
                with log.open("a", encoding="utf-8") as f:
                    f.write(f"CONTROLE STRUCTUREL (T={qn}, p=1000) : {c} positions (attendu {W + M * Lb})\n")
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


rows = []
for T in [512, 1024, 2048]:
    ids = ALL[0:T][None]
    r = {}
    CFG.update({"mode": "dense"}); r["dense"] = loss(ids)
    CFG.update({"mode": "sparse", "sel": "maxip"}); r["maxip"] = loss(ids)
    CFG.update({"sel": "pca", "dp": 8}); r["pca_d8"] = loss(ids)
    CFG.update({"sel": "rand"}); r["aleatoire"] = loss(ids)
    nb = T // Lb
    g_pca = r["pca_d8"] - r["maxip"]; g_rand = r["aleatoire"] - r["maxip"]
    rows.append((T, nb, r, g_pca, g_rand, g_pca / g_rand))
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\nT={T} (blocs candidats {nb})\n")
        f.write(f"  dense {r['dense']:.4f} | maxip {r['maxip']:.4f} | "
                f"pca_d8 {r['pca_d8']:.4f} | aleatoire {r['aleatoire']:.4f}\n")
        f.write(f"  ecart pca {g_pca:+.4f} | ecart aleatoire {g_rand:+.4f} | "
                f"rapport pca/aleatoire {g_pca / g_rand:.4f}\n")
        f.flush()
print("ok")
