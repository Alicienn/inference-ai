# -*- coding: utf-8 -*-
"""INDEX APPRIS : entrainer la base P (d' x D) par distillation de la selection maxip.
Objectif : que le classement des blocs par max (Pq).(Pk) reproduise le classement
par max q.k exact. Perte = entropie croisee entre softmax des scores exacts et
softmax des scores projetes, sur des requetes echantillonnees.
Entrainement sur le texte B, evaluation sur le texte A. Comparaison a la PCA."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M, NQ = 512, 32, 128, 2, 64
CFG = {"mode": "dense", "sel": "maxip", "P": None, "dp": 8, "seed": 0}

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
IDS_A = ALL[:T][None]
IDS_B = ALL[200000:200000 + T][None]
log = OUT / "index_appris.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} | distillation de maxip, {NQ} requetes/tete\n",
               encoding="utf-8")
COLLECT = {"on": False, "Q": [], "K": []}


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
            COLLECT["Q"].append(q[0].detach()); COLLECT["K"].append(ke[0].detach())
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


for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))


def loss(ids):
    with torch.no_grad():
        out = model(ids)
        return F.cross_entropy(out.logits[0, :-1], ids[0, 1:], reduction="none")


# --- collecte Q/K sur le texte B ---
COLLECT["on"] = True
CFG.update({"mode": "dense"})
_ = loss(IDS_B)
COLLECT["on"] = False
QS = COLLECT["Q"]; KS = COLLECT["K"]          # listes de (H, T, D)

# --- cibles exactes (maxip) sur des requetes echantillonnees ---
gen = torch.Generator().manual_seed(11)
pos = torch.randint(W + Lb, T, (NQ,), generator=gen)
TARGETS = []
for li in range(NL):
    Ql = QS[li][:, pos]                        # (H, NQ, D)
    Kl = KS[li]                                # (H, T, D)
    Se = torch.einsum("hqd,hkd->hqk", Ql, Kl) / math.sqrt(D)
    nb = T // Lb
    Sb = Se.view(H, NQ, nb, Lb)
    causb = (torch.arange(T)[None, :] <= pos[:, None])          # (NQ, T)
    Sb = Sb.masked_fill((~causb.view(NQ, nb, Lb))[None], float("-inf"))
    bvv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (pos - W + 1).clamp(min=0)[None, :])   # (nb, NQ)
    sc = Sb.max(dim=3).values.masked_fill((~bvv).T[None], float("-inf"))
    TARGETS.append(torch.softmax(sc, dim=2).detach())           # (H, NQ, nb)

# --- bases PCA de depart (texte B) ---
def pca_basis(dp, texts):
    Ps = []
    for li in range(NL):
        K = texts[li]
        C = torch.einsum("hnd,hne->hde", K, K)
        _, ev = torch.linalg.eigh(C)
        Ps.append(ev[:, :, -dp:].transpose(1, 2).detach())
    return Ps

def train_index(dp, steps=250, lr=0.02):
    P0 = pca_basis(dp, KS)
    P = torch.stack(P0).clone().requires_grad_(True)            # (NL, H, dp, D)
    opt = torch.optim.Adam([P], lr=lr)
    hist = []
    for it in range(steps):
        tot = 0.0
        opt.zero_grad()
        for li in range(NL):
            Ql = QS[li][:, pos]; Kl = KS[li]
            qp = torch.einsum("hqd,hpd->hqp", Ql, P[li])
            kp = torch.einsum("hkd,hpd->hkp", Kl, P[li])
            Sp = torch.einsum("hqp,hkp->hqk", qp, kp)
            nb = T // Lb
            Sb = Sp.view(H, NQ, nb, Lb)
            causb = (torch.arange(T)[None, :] <= pos[:, None])
            Sb = Sb.masked_fill((~causb.view(NQ, nb, Lb))[None], float("-inf"))
            sc = Sb.max(dim=3).values
            tgt = TARGETS[li]
            lg = torch.log_softmax(sc, dim=2)
            tot = tot + (-(tgt * torch.nan_to_num(lg, neginf=-50.0)).sum(dim=2)).mean()
        (tot / NL).backward()
        opt.step()
        if it % 50 == 0:
            hist.append(float(tot / NL))
    return [P[li].detach() for li in range(NL)], hist

BASES = {}
for dp in (4, 8):
    Pb, hist = train_index(dp)
    BASES[("learned", dp)] = Pb
    with log.open("a", encoding="utf-8") as f:
        f.write(f"entrainement d'={dp} : perte CE {hist[0]:.4f} -> {hist[-1]:.4f} "
                f"(warm start PCA) sur {len(hist)-1} points\n")
        f.flush()
    BASES[("pcaB", dp)] = pca_basis(dp, KS)

# --- evaluation sur le texte A ---
res = {}
CFG.update({"mode": "dense"}); res["dense"] = float(loss(IDS_A).mean())
CFG.update({"mode": "sparse", "sel": "maxip"}); res["maxip"] = float(loss(IDS_A).mean())
CFG.update({"sel": "rand", "seed": 1000}); res["aleatoire"] = float(loss(IDS_A).mean())
for dp in (4, 8):
    for tag in ("pcaB", "learned"):
        CFG.update({"sel": "proj", "P": BASES[(tag, dp)]})
        res[f"{tag}_d{dp}"] = float(loss(IDS_A).mean())
with log.open("a", encoding="utf-8") as f:
    f.write("\n=== evaluation sur le texte A (512 tokens) ===\n")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        f.write(f"{k:14s} perte={v:.4f} (p={math.exp(v):8.3f})  ecart_maxip={v - res['maxip']:+.4f}\n")
    f.write(f"\nappris d'=4 vs PCA d'=4 : {res['learned_d4'] - res['pcaB_d4']:+.4f}\n")
    f.write(f"appris d'=4 vs PCA d'=8 : {res['learned_d4'] - res['pcaB_d8']:+.4f}\n")

with log.open("a", encoding="utf-8") as f:
    f.write("\n=== sur-apprentissage : texte B (entrainement) vs texte A (tenu a l'ecart) ===\n")
    f.flush()
resB = {}
for dp in (4, 8):
    for tag in ("pcaB", "learned"):
        CFG.update({"sel": "proj", "P": BASES[(tag, dp)]})
        v = float(loss(IDS_B).mean())
        resB[f"{tag}_d{dp}"] = v
        with log.open("a", encoding="utf-8") as f:
            f.write(f"[B] {tag}_d{dp} perte={v:.4f}  |  [A] {res[f'{tag}_d{dp}']:.4f}  "
                    f"ecart B-A={v - res[f'{tag}_d{dp}']:+.4f}\n")
            f.flush()
print("ok")
