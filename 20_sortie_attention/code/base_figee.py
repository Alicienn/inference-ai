# -*- coding: utf-8 -*-
"""La base PCA est-elle figeable au prefill / transferable hors ligne ?
Variantes d'index d'=8 (12,5 % des octets de cles) :
  pca_full   : base estimee sur toute la sequence (borne haute)
  pca_f128   : base estimee sur les 128 premiers tokens puis FIGEE
  pca_f256   : idem sur 256 tokens
  pca_off    : base estimee sur un AUTRE texte (proxy apprentissage hors ligne)
Controles : maxip (cles completes), projection aleatoire, aleatoire, m0, dense."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M, DP = 512, 32, 128, 2, 8
CFG = {"mode": "dense", "sel": "maxip", "freeze": 0, "acc": [], "off": None}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
NL = cfg.num_hidden_layers
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
allids = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
IDS_A = allids[:T][None]                       # texte evalue
IDS_B = allids[T:2 * T][None]                  # texte "hors ligne"
NB = T // Lb
GEN = torch.Generator().manual_seed(7)
RP = torch.randn(DP, D, generator=GEN) / math.sqrt(DP)
log = OUT / "base_figee.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} | index d'={DP} ({100*DP//D} % des octets)\n",
               encoding="utf-8")


def loss(ids):
    with torch.no_grad():
        out = model(ids)
        return F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")


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
        if CFG["mode"] == "collect":
            ke0 = k.repeat_interleave(g, 1)[0]
            CFG["acc"][li] = torch.einsum("hnd,hne->hde", ke0, ke0).detach()
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(g > 1))
        elif CFG["mode"] == "dense":
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
                    if sel == "rproj":
                        P = RP[None].expand(H, DP, D)
                    elif sel == "offline":
                        P = CFG["off"][li]
                    elif sel.startswith("frozen"):
                        Fz = int(sel[6:])
                        C = torch.einsum("hnd,hne->hde", ke[0][:, :Fz], ke[0][:, :Fz])
                        _, ev = torch.linalg.eigh(C)
                        P = ev[:, :, -DP:].transpose(1, 2)
                    elif sel == "pca":
                        C = torch.einsum("hnd,hne->hde", ke[0], ke[0])
                        _, ev = torch.linalg.eigh(C)
                        P = ev[:, :, -DP:].transpose(1, 2)
                    if sel in ("rproj", "offline") or sel.startswith("frozen") or sel == "pca":
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


for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))

# bases "hors ligne" : second moment accumule sur un AUTRE texte
CFG.update({"mode": "collect", "acc": [None] * NL})
_ = loss(IDS_B)
off = []
for li in range(NL):
    _, ev = torch.linalg.eigh(CFG["acc"][li])
    off.append(ev[:, :, -DP:].transpose(1, 2))
CFG["off"] = off

CFG.update({"mode": "m0"})
d = float((loss(IDS_A)[:255] - loss(IDS_A[:, :256])).abs().max())
with log.open("a", encoding="utf-8") as f:
    f.write(f"AUTO-TEST CAUSAL {d:.3e} -> {'OK' if d < 1e-3 else 'ECHEC'}\n")

res = {}
def R(k, v):
    res[k] = v
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{k:20s} perte={v:.4f}\n")
        f.flush()
CFG.update({"mode": "dense"}); R("dense", float(loss(IDS_A).mean()))
CFG.update({"mode": "m0"}); R("m0_fenetre", float(loss(IDS_A).mean()))
CFG.update({"mode": "sparse", "sel": "maxip"}); R("maxip_D64", float(loss(IDS_A).mean()))
CFG.update({"sel": "oracle"}); R("oracle_LSE", float(loss(IDS_A).mean()))
CFG.update({"sel": "pca"}); R("pca_full_d8", float(loss(IDS_A).mean()))
CFG.update({"sel": "frozen128"}); R("pca_frozen128_d8", float(loss(IDS_A).mean()))
CFG.update({"sel": "frozen256"}); R("pca_frozen256_d8", float(loss(IDS_A).mean()))
CFG.update({"sel": "offline"}); R("pca_offline_d8", float(loss(IDS_A).mean()))
CFG.update({"sel": "rproj"}); R("randproj_d8", float(loss(IDS_A).mean()))
CFG.update({"sel": "rand"})
rands = []
for sd in (1000, 2000):
    CFG["seed"] = sd
    rands.append(float(loss(IDS_A).mean()))
R("rand_moy", statistics.mean(rands))
with log.open("a", encoding="utf-8") as f:
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        f.write(f"{k:20s} perte={v:.4f} (p={math.exp(v):8.3f})\n")
    f.write(f"\necart a maxip : full {res['pca_full_d8']-res['maxip_D64']:+.4f} ; "
            f"frozen128 {res['pca_frozen128_d8']-res['maxip_D64']:+.4f} ; "
            f"frozen256 {res['pca_frozen256_d8']-res['maxip_D64']:+.4f} ; "
            f"offline {res['pca_offline_d8']-res['maxip_D64']:+.4f}\n")
    f.write(f"bat le hasard ? full={res['pca_full_d8']<min(rands)} "
            f"frozen128={res['pca_frozen128_d8']<min(rands)} frozen256={res['pca_frozen256_d8']<min(rands)} "
            f"offline={res['pca_offline_d8']<min(rands)} rproj={res['randproj_d8']<min(rands)}\n")
print("ok")
