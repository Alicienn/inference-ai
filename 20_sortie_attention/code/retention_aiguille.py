# -*- coding: utf-8 -*-
"""RETENTION DU BLOC D'AIGUILLE (mesure non degeneree). Pour des requetes en fin de
sequence, on mesure la masse d'attention dense portee aux clefs de l'aiguille (pos 48-67)
et la masse retenue par la selection creuse, plus la fraction de tetes/requetes ou le
bloc de l'aiguille est effectivement selectionne."""
import math, pathlib, torch, torch.nn.functional as F, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, NBASE, DP, LB, NT = 512, 256, 8, 4, 8
NLO, NHI = 48, 68
POS = list(range(T - 16, T - 1))
CFG = {"mode": "dense", "W": 64, "m": 32}
BASE = {}
ACC = {"md": 0.0, "ms": 0.0, "sel": 0, "n": 0, "ms_ow": 0.0}
FRACS = [0.375, 0.125, 0.0625]

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]


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
        ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
        h = q.shape[1]
        if CFG["mode"] == "collect":
            K = ke[0].float(); Kc = K - K.mean(1, keepdim=True)
            U, _, _ = torch.linalg.svd(Kc.transpose(1, 2), full_matrices=False)
            BASE[li] = U[:, :, :DP].contiguous()
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2)
        del S
        W, m = CFG["W"], CFG["m"]
        if CFG["mode"] == "pca" and m > 0 and qn == T:
            nb = qn // LB
            U = BASE[li]
            qp = torch.einsum("hqd,hde->hqe", q[0], U)
            kp = torch.einsum("hkd,hde->hke", ke[0], U)
            sc = (qp[:, :, None, :] * kp[:, None, :, :]).amax(-1)
            B = sc[:, :, :nb * LB].view(h, qn, nb, LB).amax(-1)
            del qp, kp, sc
            bid = torch.arange(nb)
            elig = (bid[None, :] * LB + LB - 1) <= (torch.arange(qn) - W)[:, None]
            _, sel = torch.topk(B * elig[None], min(m, nb), dim=2)
            keys = (sel[..., None] * LB + torch.arange(LB)).reshape(h, qn, -1)
            mask = torch.zeros(h, qn, qn, dtype=torch.bool)
            mask.scatter_(2, keys, True)
            mask |= (j[None, None, :] > (j[None, :, None] - W))
            md = A[:, POS, NLO:NHI].sum(-1)
            sp = A * mask
            ms = sp[:, POS, NLO:NHI].sum(-1)
            ow = sp[:, POS].sum(-1)
            ACC["md"] += float(md.mean()); ACC["ms"] += float(ms.mean())
            ACC["ms_ow"] += float((ms / ow.clamp(min=1e-9)).mean())
            ACC["sel"] += int(mask[:, POS, NLO:NHI].any(-1).sum())
            ACC["n"] += mask[:, POS, NLO:NHI].any(-1).numel()
            A = sp / sp.sum(-1, keepdim=True).clamp(min=1e-9)
            del mask, B, sp
        o = torch.einsum("hqk,hkd->hqd", A, ve[0])
        del A
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))

CFG["mode"] = "collect"
with torch.no_grad():
    model(ALL[:NBASE][None])
print("base collectee", flush=True)

seqs = []
rng = random.Random(7)
for i in range(NT):
    s = rng.randint(2000, len(ALL) - T - 32)
    needle = ALL[s:s + 16].clone()
    seq = ALL[0:T].clone()
    seq[50:66] = needle
    seq[T - 16:T] = needle
    seqs.append(seq)

log = OUT / "retention_aiguille.txt"
log.write_text(f"{MID} | T={T} | aiguille 16 tok en 50-65 | clefs mesurees {NLO}-{NHI-1} | "
               f"requetes {len(POS)} | index PCA d'={DP} max | 30 couches x {NT} seqs\n"
               "config       W    m    masse_dense  masse_retenue  part_du_OW  bloc_sel%\n",
               encoding="utf-8")


def run(frac):
    if frac is None:
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
        md = 0.0; n = 0
        for seq in seqs:
            CFG["mode"] = "probe_dense"
            with torch.no_grad():
                model(seq[None])
        return None
    CFG["mode"] = "pca"
    CFG["W"] = max(8, int(T * frac / 3)); CFG["m"] = max(1, int((T * frac * 2 / 3) // LB))
    for k in ACC:
        ACC[k] = 0
    for seq in seqs:
        with torch.no_grad():
            model(seq[None])
    md = ACC["md"] / NT; ms = ACC["ms"] / NT
    row = (CFG["W"], CFG["m"], md, ms, ACC["ms_ow"] / NT, 100.0 * ACC["sel"] / max(ACC["n"], 1))
    with log.open("a", encoding="utf-8") as f:
        f.write(f"pca {frac:.3f}  {row[0]:4d} {row[1]:4d}  {row[2]:.3e}  {row[3]:.3e}  "
                f"{row[4]:.4f}  {row[5]:.1f}\n")
    print(f"frac={frac:.3f} W={row[0]} m={row[1]} masse_dense={row[2]:.3e} "
          f"retenue={row[3]:.3e} part_OW={row[4]:.4f} bloc_sel={row[5]:.1f}%", flush=True)


for fr in FRACS:
    run(fr)
print("FIN")
