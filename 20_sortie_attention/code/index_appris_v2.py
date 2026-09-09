# -*- coding: utf-8 -*-
"""Index appris, version corrigee : base ORTHONORMEE (QR) + beaucoup plus de donnees.
Question : l'echec precedent venait-il du sur-apprentissage ou de l'objectif ?
4 textes d'entrainement, 128 requetes/tete, 150 pas, P sur la variete de Stiefel."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M, NQ = 512, 32, 128, 2, 128
TRAIN_OFF = [100000, 150000, 200000, 250000]
CFG = {"mode": "dense", "sel": "maxip", "P": None, "dp": 4, "seed": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
NL = cfg.num_hidden_layers
NB = T // Lb
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
log = OUT / "index_appris_v2.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} | base orthonormee, {NQ} req/tete, "
               f"{len(TRAIN_OFF)} textes d'entrainement\n", encoding="utf-8")
COLLECT = {"on": False, "Q": None, "K": []}


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
        ke = k.repeat_interleave(g, 1)
        if COLLECT["on"]:
            COLLECT["K"].append(ke[0].detach())
            COLLECT["Q"] = COLLECT["Q"] + [q[0][:, POS].detach()]
        if CFG["mode"] == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(g > 1))
        else:
            ve = v.repeat_interleave(g, 1)
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
                if sel == "maxip":
                    Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                else:
                    P = CFG["P"][li]
                    qp = torch.einsum("hqd,hpd->hqp", q[0], P)
                    kp = torch.einsum("hkd,hpd->hkp", ke[0], P)
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


gen = torch.Generator().manual_seed(11)
POS = torch.randint(W + Lb, T, (NQ,), generator=gen)
for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))


def loss(ids):
    with torch.no_grad():
        out = model(ids)
        return F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")


CHUNKS = []
for off in TRAIN_OFF:
    COLLECT.update({"on": True, "Q": [], "K": []})
    CFG.update({"mode": "dense"})
    _ = loss(ALL[off:off + T][None])
    COLLECT["on"] = False
    CHUNKS.append({"Q": COLLECT["Q"], "K": COLLECT["K"]})

causb = (torch.arange(T)[None, :] <= POS[:, None]).view(NQ, NB, Lb)
bvv = ((torch.arange(NB) * Lb + Lb)[:, None] <= (POS - W + 1).clamp(min=0)[None, :])


def targets(Q, K):
    out = []
    for li in range(NL):
        Se = torch.einsum("hqd,hkd->hqk", Q[li], K[li]) / math.sqrt(D)
        Sb = Se.view(H, NQ, NB, Lb).masked_fill((~causb)[None], float("-inf"))
        sc = Sb.max(dim=3).values.masked_fill((~bvv).T[None], float("-inf"))
        out.append(torch.softmax(sc, dim=2).detach())
    return out


TG = [targets(c["Q"], c["K"]) for c in CHUNKS]


def pca_basis(dp, K):
    Ps = []
    for li in range(NL):
        C = torch.einsum("hnd,hne->hde", K[li], K[li])
        _, ev = torch.linalg.eigh(C)
        Ps.append(ev[:, :, -dp:].transpose(1, 2).detach())
    return Ps


def orth(Praw):
    Qm, _ = torch.linalg.qr(Praw.transpose(2, 3))     # (NL,H,D,dp) : QR sur (D,dp)
    return Qm.transpose(2, 3)                          # (NL,H,dp,D) : lignes orthonormees


def train(dp, steps=150, lr=0.05):
    P0 = torch.stack(pca_basis(dp, CHUNKS[0]["K"]))
    Praw = P0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Praw], lr=lr)
    hist = []
    for it in range(steps):
        P = orth(Praw)
        tot = 0.0
        for c, ch in enumerate(CHUNKS):
            for li in range(NL):
                qp = torch.einsum("hqd,hpd->hqp", ch["Q"][li], P[li])
                kp = torch.einsum("hkd,hpd->hkp", ch["K"][li], P[li])
                Sp = torch.einsum("hqp,hkp->hqk", qp, kp)
                Sb = Sp.view(H, NQ, NB, Lb).masked_fill((~causb)[None], float("-inf"))
                sc = Sb.max(dim=3).values
                lg = torch.log_softmax(sc, dim=2)
                tot = tot + (-(TG[c][li] * torch.nan_to_num(lg, neginf=-50.0)).sum(dim=2)).mean()
        lossv = tot / (len(CHUNKS) * NL)
        opt.zero_grad(); lossv.backward(); opt.step()
        if it % 30 == 0:
            hist.append(float(lossv))
    return [orth(Praw)[li].detach() for li in range(NL)], hist


LEARNED, HIST = train(CFG["dp"])
with log.open("a", encoding="utf-8") as f:
    f.write(f"CE entrainement (d'={CFG['dp']}, base orthonormee) : "
            f"{HIST[0]:.4f} -> {HIST[-1]:.4f} sur {len(HIST)} points\n")
    f.flush()

PCAB4 = pca_basis(4, CHUNKS[2]["K"])
PCAB8 = pca_basis(8, CHUNKS[2]["K"])

res = {}
IDS_A = ALL[:T][None]
CFG.update({"mode": "dense"}); res["dense"] = float(loss(IDS_A).mean())
CFG.update({"mode": "sparse", "sel": "maxip"}); res["maxip"] = float(loss(IDS_A).mean())
CFG.update({"sel": "rand", "seed": 1000}); res["aleatoire"] = float(loss(IDS_A).mean())
for tag, P in (("pca_d4", PCAB4), ("pca_d8", PCAB8), ("appris_d4", LEARNED)):
    CFG.update({"sel": "proj", "P": P}); res[tag] = float(loss(IDS_A).mean())
with log.open("a", encoding="utf-8") as f:
    f.write("\n=== texte A (tenu a l'ecart) ===\n")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        f.write(f"{k:12s} perte={v:.4f}  ecart_maxip={v - res['maxip']:+.4f}\n")
    f.write(f"appris_d4 vs pca_d4 : {res['appris_d4'] - res['pca_d4']:+.4f}\n")
    f.write(f"appris_d4 vs pca_d8 : {res['appris_d4'] - res['pca_d8']:+.4f}\n")
    f.flush()

# controle in-sample : texte d'entrainement
IDS_B = ALL[TRAIN_OFF[2]:TRAIN_OFF[2] + T][None]
with log.open("a", encoding="utf-8") as f:
    f.write("\n=== texte d'entrainement (in-sample) ===\n")
    f.flush()
for tag, P in (("pca_d4", PCAB4), ("appris_d4", LEARNED)):
    CFG.update({"sel": "proj", "P": P})
    v = float(loss(IDS_B).mean())
    with log.open("a", encoding="utf-8") as f:
        f.write(f"[B] {tag:12s} perte={v:.4f}\n")
        f.flush()
print("ok")
