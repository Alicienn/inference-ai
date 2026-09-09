# -*- coding: utf-8 -*-
"""Transfert inter-domaines de la base de l'index compact (d'=8, 12,5 % des octets).
Trois domaines : corpus PDF academique | wikitext | code Python local.
Pour chaque domaine d : base estimee sur les 512 premiers tokens, eval sur les 512 suivants.
Matrice 3x3 des croisements base x texte d'evaluation. Controles : dense, maxip, aleatoire."""
import math, pathlib, statistics, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M, DP = 512, 32, 128, 2, 8
CFG = {"mode": "dense", "sel": "maxip", "P": None, "acc": [], "seed": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
NL = cfg.num_hidden_layers
NB = T // Lb
log = OUT / "transfert_domaines.txt"
log.write_text(f"{MID} | index d'={DP} ({100*DP//D} % des octets) | T={T} W={W} Lb={Lb} m={M}\n",
               encoding="utf-8")

corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
code = ""
for f in sorted((ROOT / "20_sortie_attention" / "code").glob("*.py")):
    code += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
wt = ""
for r in ds:
    wt += r["text"]
    if len(wt) > 200000:
        break
TEXTS = {"corpus": corpus, "wikitext": wt, "code": code}
IDS = {}
for k, v in TEXTS.items():
    ids = tok(v, return_tensors="pt", add_special_tokens=False).input_ids[0]
    IDS[k] = (ids[:T][None], ids[T:2 * T][None])   # (base, eval)


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


BASES = {}
for d, (base_ids, _) in IDS.items():
    CFG.update({"mode": "collect", "acc": [None] * NL})
    _ = loss(base_ids)
    P = []
    for li in range(NL):
        _, ev = torch.linalg.eigh(CFG["acc"][li])
        P.append(ev[:, :, -DP:].transpose(1, 2))
    BASES[d] = P
    with log.open("a", encoding="utf-8") as f:
        f.write(f"base {d:9s} estimee sur 512 tokens ({base_ids.shape[1]} -> ok)\n")
        f.flush()


def R(k, v):
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{k:26s} perte={v:.4f} (p={math.exp(v):8.3f})\n")
        f.flush()


for ev, (_, ev_ids) in IDS.items():
    CFG.update({"mode": "dense"}); R(f"[{ev}] dense", float(loss(ev_ids).mean()))
    CFG.update({"mode": "sparse", "sel": "maxip"}); R(f"[{ev}] maxip", float(loss(ev_ids).mean()))
    CFG.update({"sel": "rand"})
    rr = []
    for sd in (1000, 2000):
        CFG["seed"] = sd; rr.append(float(loss(ev_ids).mean()))
    R(f"[{ev}] aleatoire", statistics.mean(rr))
    for bd in ("corpus", "wikitext", "code"):
        CFG.update({"sel": "basis", "P": BASES[bd]})
        R(f"[{ev}] base_{bd}", float(loss(ev_ids).mean()))
print("ok")
