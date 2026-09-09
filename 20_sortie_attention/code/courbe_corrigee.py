# -*- coding: utf-8 -*-
"""Harnais CORRIGE (bug du sentinelle -1) + courbe octets-qualite remesuree.
Bug : ti.clamp(min=0) ramenait l'indice invalide -1 a 0, marquant le bloc 0
(qui contient le sink) comme selectionne des qu'il y avait moins de m blocs valides.
Correction : n'ajouter que les selections valides."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M = 512, 32, 128, 2
OFFSETS = [0, 200000, 400000]
CFG = {"mode": "dense", "sel": "maxip", "dp": 8, "seed": 0, "check": True}

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
log = OUT / "courbe_corrigee3.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} | HARNAIS CORRIGE (sentinelle -1)\n", encoding="utf-8")


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
                gen = torch.Generator().manual_seed(CFG["seed"])
                ti = torch.randint(0, nb, (H, qn, M), generator=gen)
                ok = bv.T[None].expand(H, qn, nb).gather(2, ti.clamp(max=nb - 1))
                ti = torch.where(ok, ti, torch.full_like(ti, -1))
            else:
                if sel in ("maxip", "oracle"):
                    Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                else:
                    dp = CFG["dp"]
                    K = ke[0]
                    C = torch.einsum("hnd,hne->hde", K, K)
                    _, ev = torch.linalg.eigh(C)
                    P = ev[:, :, -dp:].transpose(1, 2)
                    qp = torch.einsum("hqd,hpd->hqp", q[0], P)
                    kp = torch.einsum("hkd,hpd->hkp", K, P)
                    Sp = torch.einsum("hqp,hkp->hqk", qp, kp)
                Sb = Sp[:, :, :nb * Lb].view(H, qn, nb, Lb)
                causb = causal[:, :nb * Lb].view(qn, nb, Lb)
                Sb = Sb.masked_fill((~causb)[None], float("-inf"))
                if sel == "oracle":
                    sc = torch.logsumexp(Sb, dim=3)
                else:
                    sc = Sb.max(dim=3).values
                sc = sc.masked_fill((~bv).T[None], float("-inf"))
                tv, ti = torch.topk(sc, M, dim=2)
                ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
            # --- CORRECTION : ne marquer que les selections valides ---
            valid = ti >= 0
            onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
            hh, pp, _ = torch.nonzero(valid, as_tuple=True)
            onehot[hh, pp, ti[valid]] = True
            allowed |= onehot[:, :, (j // Lb)] & causal[None]
            if CFG["check"] and qn == T:
                c100 = int(allowed[0, 100].sum()); c300 = int(allowed[0, 300].sum())
                with log.open("a", encoding="utf-8") as f:
                    f.write(f"CONTROLE STRUCTUREL : p=100 -> {c100} (attendu 101, fenetre seule) ; "
                            f"p=300 -> {c300} (attendu {W + M * Lb})\n")
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
        return F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")


rows = {}
for off in OFFSETS:
    ids = ALL[off:off + T][None]
    CFG.update({"mode": "sparse", "sel": "maxip"})
    dt = float((loss(ids)[:255] - loss(ids[:, :256])).abs().max())
    res = {}
    CFG.update({"mode": "dense"}); res["dense"] = float(loss(ids).mean())
    CFG.update({"mode": "sparse", "sel": "maxip"}); res["maxip"] = float(loss(ids).mean())
    CFG.update({"sel": "oracle"}); res["oracle"] = float(loss(ids).mean())
    CFG.update({"sel": "pca", "dp": 16}); res["pca_d16"] = float(loss(ids).mean())
    CFG.update({"sel": "pca", "dp": 8}); res["pca_d8"] = float(loss(ids).mean())
    CFG.update({"sel": "rand", "seed": 1000}); res["aleatoire"] = float(loss(ids).mean())
    rows[off] = res
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\noffset {off} (auto-test causal {dt:.1e})\n")
        for k, v in sorted(res.items(), key=lambda kv: kv[1]):
            f.write(f"  {k:10s} {v:.4f}\n")
        f.flush()

with log.open("a", encoding="utf-8") as f:
    f.write("\n=== moyennes sur 3 tranches (harnais corrige) ===\n")
    m = {k: statistics.mean([rows[o][k] for o in OFFSETS]) for k in ["dense", "maxip", "oracle", "pca_d16", "pca_d8", "aleatoire"]}
    for k, v in sorted(m.items(), key=lambda kv: kv[1]):
        f.write(f"{k:10s} moy={v:.4f}  ecart_maxip={v - m['maxip']:+.4f}\n")
print("ok")
