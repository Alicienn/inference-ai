# -*- coding: utf-8 -*-
"""Regime a plus grand nombre de blocs candidats : T=1024, W=128, Lb=32, m=2
=> (1024-128)/32 = 28 blocs candidats (contre 12 a T=512).
Variantes : dense | maxip | aleatoire | PCA d'=8 (12,5 % des octets). 2 tranches."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M = 1024, 32, 128, 2
OFFSETS = [0, 200000]
CFG = {"mode": "dense", "sel": "maxip", "dp": 8, "seed": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
NB = T // Lb
NCAND = NB - (W // Lb)
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
log = OUT / "contexte_long.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} | blocs candidats ~{NCAND}\n", encoding="utf-8")


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
            if CFG["mode"] != "m0":
                sel = CFG["sel"]
                if sel == "rand":
                    gen = torch.Generator().manual_seed(CFG["seed"])
                    ti = torch.randint(0, nb, (H, qn, M), generator=gen)
                    ok = bv.T[None].expand(H, qn, nb).gather(2, ti.clamp(max=nb - 1))
                    ti = torch.where(ok, ti, torch.full_like(ti, -1))
                else:
                    if sel == "maxip":
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
                    sc = Sb.max(dim=3).values
                    sc = sc.masked_fill((~bv).T[None], float("-inf"))
                    tv, ti = torch.topk(sc, M, dim=2)
                    ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
                onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
                _vv = ti >= 0
                _hh, _pp, _ = torch.nonzero(_vv, as_tuple=True)
                onehot[_hh, _pp, ti[_vv]] = True
                allowed |= onehot[:, :, (j // Lb)] & causal[None]
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
    CFG.update({"mode": "m0"})
    dt = float((loss(ids)[:255] - loss(ids[:, :256])).abs().max())
    res = {}
    CFG.update({"mode": "dense"}); res["dense"] = float(loss(ids).mean())
    CFG.update({"mode": "sparse", "sel": "maxip"}); res["maxip"] = float(loss(ids).mean())
    CFG.update({"sel": "rand", "seed": 1000}); res["aleatoire"] = float(loss(ids).mean())
    CFG.update({"sel": "pca", "dp": 8}); res["pca_d8"] = float(loss(ids).mean())
    rows[off] = res
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\noffset {off} (auto-test causal {dt:.1e})\n")
        for k, v in sorted(res.items(), key=lambda kv: kv[1]):
            f.write(f"  {k:10s} {v:.4f}\n")
        f.flush()

with log.open("a", encoding="utf-8") as f:
    f.write("\n=== moyennes sur 2 tranches (T=1024) ===\n")
    for k in ["dense", "maxip", "pca_d8", "aleatoire"]:
        vals = [rows[o][k] for o in OFFSETS]
        f.write(f"{k:10s} moy={statistics.mean(vals):.4f}\n")
    f.write(f"ecart a maxip : pca_d8 {statistics.mean([rows[o]['pca_d8'] for o in OFFSETS]) - statistics.mean([rows[o]['maxip'] for o in OFFSETS]):+.4f} ; "
            f"aleatoire {statistics.mean([rows[o]['aleatoire'] for o in OFFSETS]) - statistics.mean([rows[o]['maxip'] for o in OFFSETS]):+.4f}\n")
print("ok")
