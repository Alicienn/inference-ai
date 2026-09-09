# -*- coding: utf-8 -*-
"""Verification independante (v2) : chemin explicite vs chemin patche.
Compare selection, ensemble autorise, sortie d'attention brute et sortie finale,
en erreur absolue ET relative."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, M, P0 = 512, 32, 128, 2, 300
LAYERS = [0, 29]
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
cfg = model.config
H = cfg.num_attention_heads; Hkv = cfg.num_key_value_heads
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
IDS = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0][:T][None]
CAP = {}


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
        nb = qn // Lb
        j = torch.arange(qn); p = torch.arange(qn)
        causal = (j[None, :] <= p[:, None])
        allowed = ((j[None, :] >= (p - W + 1).clamp(min=0)[:, None]) & causal)[None].expand(H, qn, qn).clone()
        bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
        Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
        Sb = Sp[:, :, :nb * Lb].view(H, qn, nb, Lb)
        causb = causal[:, :nb * Lb].view(qn, nb, Lb)
        Sb = Sb.masked_fill((~causb)[None], float("-inf"))
        sc = Sb.max(dim=3).values.masked_fill((~bv).T[None], float("-inf"))
        tv, ti = torch.topk(sc, M, dim=2)
        ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
        onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
        _vv = ti >= 0
        _hh, _pp, _ = torch.nonzero(_vv, as_tuple=True)
        onehot[_hh, _pp, ti[_vv]] = True
        allowed |= onehot[:, :, (j // Lb)] & causal[None]
        o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        if li in LAYERS:
            CAP[li] = {"in": hidden_states.detach(), "pos": (cos.detach(), sin.detach()),
                       "raw": o.detach(), "out": self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)).detach(),
                       "ti": ti[:, P0].clone(), "row": allowed[0, P0].clone()}
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))
with torch.no_grad():
    _ = model(IDS)


def reference(li):
    att = model.model.layers[li].self_attn
    h_in = CAP[li]["in"]; cos, sin = CAP[li]["pos"]
    q = att.q_proj(h_in).view(1, T, H, D).transpose(1, 2)
    k = att.k_proj(h_in).view(1, T, Hkv, D).transpose(1, 2)
    v = att.v_proj(h_in).view(1, T, Hkv, D).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    g = att.num_key_value_groups
    ke = k.repeat_interleave(g, 1)[0]; ve = v.repeat_interleave(g, 1)[0]
    q0 = q[0]; nb = T // Lb
    out = torch.zeros(H, T, D)
    sel_at_p0 = None; row_at_p0 = None
    for p in range(T):
        lo = max(0, p - W + 1)
        blk = [b for b in range(nb) if (b + 1) * Lb <= lo]
        if blk:
            sc = torch.stack([(q0[:, p].unsqueeze(1) * ke[:, b * Lb:(b + 1) * Lb]).sum(-1).max(-1).values
                              for b in blk], dim=1)
            top = sc.topk(min(M, len(blk)), dim=1).indices
        else:
            top = torch.zeros(H, 0, dtype=torch.long)
        for h in range(H):
            allowed = set(range(lo, p + 1))
            if blk:
                for t in top[h].tolist():
                    b = blk[t]
                    allowed |= set(range(b * Lb, (b + 1) * Lb))
            idx = torch.tensor(sorted(allowed), dtype=torch.long)
            lg = (q0[h, p] @ ke[h, idx].T) / math.sqrt(D)
            out[h, p] = torch.softmax(lg, dim=0) @ ve[h, idx]
        if p == P0:
            row_at_p0 = allowed
            sel_at_p0 = [[blk[t] for t in top[h].tolist()] for h in range(H)] if blk else [[]] * H
    raw = out.unsqueeze(0)
    fin = att.o_proj(raw.transpose(1, 2).reshape(1, T, -1))
    return raw, fin, sel_at_p0, row_at_p0


log = OUT / "verification_independante.txt"
log.write_text(f"{MID} | T={T} W={W} Lb={Lb} m={M} | chemin independant (boucles explicites) vs patche (SDPA)\n", encoding="utf-8")
for li in LAYERS:
    raw, fin, sel, row = reference(li)
    a_raw = float((raw - CAP[li]["raw"]).abs().max()); n_raw = float(CAP[li]["raw"].abs().max())
    a_fin = float((fin - CAP[li]["out"]).abs().max()); n_fin = float(CAP[li]["out"].abs().max())
    pat_row = set(torch.nonzero(CAP[li]["row"]).flatten().tolist())
    pat_sel = [[int(x) for x in CAP[li]["ti"][h].tolist() if int(x) >= 0] for h in range(H)]
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\ncouche {li} :\n")
        f.write(f"  selection identique (toutes tetes, p={P0}) : {pat_sel == sel}\n")
        f.write(f"  ensemble autorise identique (p={P0}) : {pat_row == row} "
                f"({len(pat_row)} positions patchees, {len(row)} reference)\n")
        f.write(f"  sortie d'attention brute : abs={a_raw:.3e}  rel={a_raw / n_raw:.3e}\n")
        f.write(f"  sortie apres o_proj     : abs={a_fin:.3e}  rel={a_fin / n_fin:.3e}\n")
        f.flush()
print("ok")
