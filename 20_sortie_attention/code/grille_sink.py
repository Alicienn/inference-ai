# -*- coding: utf-8 -*-
"""GRILLE (T,W) : la decomposition sink+fond tient-elle quand le contexte grandit ?
Question decisive : a T grand, l'argmax hors fenetre sort-il du debut de sequence
(=> composante de RECUPERATION, que le selecteur doit trouver) ?"""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
GRID = [(512, 128), (1024, 128), (2048, 128), (2048, 256)]
CFG = {"W": 128}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
ACC = {}


def stats(A, qn):
    W = CFG["W"]
    pmin = W + 64
    if pmin >= qn - 1:
        return
    idx = torch.arange(pmin, qn)
    L = (idx - W + 1).float()                       # taille hors fenetre
    M = (torch.arange(qn)[None, :] <= (idx - W)[:, None])
    Asub = A[pmin:qn] * M
    out = Asub.sum(1)
    cond = Asub / out.clamp(min=1e-12)[:, None]
    am = cond.argmax(1)
    ent = -(cond * (cond + 1e-12).log()).sum(1)
    top1 = cond.max(1).values
    dec_pos = (10.0 * am.float() / L).long().clamp(0, 9)
    D = (10.0 * torch.arange(qn)[None, :].float() / L[:, None]).long().clamp(0, 9)
    prof = torch.zeros(10)
    for b in range(10):
        prof[b] = float((cond * ((D == b) & M)).sum())
    k = (qn, W)
    a = ACC.setdefault(k, {"n": 0, "out": 0.0, "eff": 0.0, "top1": 0.0, "f10": 0, "f32": 0,
                           "prof": torch.zeros(10), "amdec": torch.zeros(10)})
    n = len(idx)
    a["n"] += n
    a["out"] += float(out.sum()); a["eff"] += float(torch.exp(ent).sum())
    a["top1"] += float(top1.sum())
    a["f10"] += int((am < 10).sum()); a["f32"] += int((am < 32).sum())
    a["prof"] += prof
    a["amdec"] += torch.bincount(dec_pos, minlength=10).float()


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
        ke = k.repeat_interleave(g, 1)
        o = F.scaled_dot_product_attention(q, ke, v.repeat_interleave(g, 1), is_causal=True)
        S = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(self.head_dim)
        j = torch.arange(qn)
        S = S.masked_fill((j[None, None, :] > j[None, :, None]), float("-inf"))
        A = torch.softmax(S, dim=2).mean(dim=0)
        del S
        stats(A, qn)
        del A
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))

log = OUT / "grille_sink.txt"
log.write_text(f"{MID} | grille (T,W) | offset 0\n"
               "T     W    out_mass  eff   top1   f10    f32    argmax_par_decile(1..10)\n",
               encoding="utf-8")
for (T, W) in GRID:
    CFG["W"] = W
    with torch.no_grad():
        model(ALL[:T][None])
    a = ACC.get((T, W))
    if not a:
        print(f"T={T} W={W} : pas de donnees", flush=True); continue
    n = a["n"]
    amd = (a["amdec"] / n).tolist()
    prof = (a["prof"] / n).tolist()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{T:5d} {W:4d} {a['out']/n:8.4f} {a['eff']/n:6.2f} {a['top1']/n:6.4f} "
                f"{a['f10']/n:6.4f} {a['f32']/n:6.4f} "
                f"{' '.join(f'{v:.3f}' for v in amd)}\n")
        f.write(f"      profil distance : {' '.join(f'{v:.4f}' for v in prof)}\n")
    print(f"T={T} W={W} out={a['out']/n:.4f} eff={a['eff']/n:.2f} top1={a['top1']/n:.4f} "
          f"f10={a['f10']/n:.4f} f32={a['f32']/n:.4f}", flush=True)
    print(f"   argmax_deciles={[round(v,3) for v in amd]}", flush=True)
    print(f"   profil={[round(v,4) for v in prof]}", flush=True)
print("FIN")
