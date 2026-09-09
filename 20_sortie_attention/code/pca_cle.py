# -*- coding: utf-8 -*-
"""Derniere famille de compression non refutee : base DEPENDANTE DES DONNEES.
On projette les cles sur les d' directions principales (PCA par tete, calculable au prefill),
puis score = max (Pq).(Pk). Comparer a la projection aleatoire, a maxip et au hasard."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M = 512, 32, 128, 2
CFG = {"mode": "dense", "sel": "pca", "dp": 16, "seed": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
IDS = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0, :T][None]
NB = T // Lb
log = OUT / "pca_cle.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} | PCA par tete (donnees)\n", encoding="utf-8")
GEN = torch.Generator().manual_seed(7)
RP = {}


def rproj(dp):
    if dp not in RP:
        RP[dp] = torch.randn(dp, D, generator=GEN) / math.sqrt(dp)
    return RP[dp]


def loss(ids=IDS):
    with torch.no_grad():
        out = model(ids)
        return F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")


def patched():
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
                    if sel in ("pca", "rproj"):
                        dp = CFG["dp"]
                        if sel == "pca":
                            K = ke[0]                                   # (H, qn, D)
                            C = torch.einsum("hnd,hne->hde", K, K) / qn  # second moment
                            _, evec = torch.linalg.eigh(C)
                            P = evec[:, :, -dp:].transpose(1, 2)         # (H, dp, D)
                        else:
                            P = rproj(dp)[None].expand(H, dp, D)
                        qp = torch.einsum("hqd,hpd->hqp", q[0], P)
                        kp = torch.einsum("hkd,hpd->hkp", ke[0], P)
                        Sp = torch.einsum("hqp,hkp->hqk", qp, kp)
                    else:
                        Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                    Sb = Sp[:, :, :nb * Lb].view(H, qn, nb, Lb)
                    causb = causal[:, :nb * Lb].view(qn, nb, Lb)
                    Sb = Sb.masked_fill((~causb)[None], float("-inf"))
                    sc = Sb.max(dim=3).values if sel != "oracle" else torch.logsumexp(Sb, dim=3)
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
    layer.self_attn.forward = patched().__get__(layer.self_attn, type(layer.self_attn))

CFG.update({"mode": "m0"})
d = float((loss(IDS)[:255] - loss(IDS[:, :256])).abs().max())
with log.open("a", encoding="utf-8") as f:
    f.write(f"AUTO-TEST CAUSAL {d:.3e} -> {'OK' if d < 1e-3 else 'ECHEC'}\n")

res = {}
CFG.update({"mode": "dense"}); res["dense"] = float(loss().mean())
CFG.update({"mode": "m0"}); res["m0_fenetre"] = float(loss().mean())
CFG.update({"mode": "sparse", "sel": "maxip"}); res["maxip_D64"] = float(loss().mean())
CFG.update({"sel": "oracle"}); res["oracle_LSE"] = float(loss().mean())
for dp in (8, 16, 32):
    CFG.update({"sel": "pca", "dp": dp}); res[f"PCA_d{dp}"] = float(loss().mean())
    CFG.update({"sel": "rproj", "dp": dp}); res[f"randproj_d{dp}"] = float(loss().mean())
CFG.update({"sel": "rand"})
rands = []
for sd in (1000, 2000):
    CFG["seed"] = sd
    rands.append(float(loss().mean()))
res["rand_moy"] = statistics.mean(rands)
with log.open("a", encoding="utf-8") as f:
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        f.write(f"{k:16s} perte={v:.4f} (p={math.exp(v):8.3f})\n")
    f.write(f"\nPCA vs projection aleatoire (meme budget) :\n")
    for dp in (8, 16, 32):
        f.write(f"  d'={dp:2d} ({100*dp//D}% octets) : PCA {res[f'PCA_d{dp}']:.4f} vs "
                f"alea {res[f'randproj_d{dp}']:.4f} -> ecart {res[f'PCA_d{dp}'] - res[f'randproj_d{dp}']:+.4f}\n")
    f.write(f"PCA bat le hasard ? d8={res['PCA_d8'] < min(rands)} d16={res['PCA_d16'] < min(rands)} "
            f"d32={res['PCA_d32'] < min(rands)}\n")
print("ok")
