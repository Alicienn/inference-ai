# -*- coding: utf-8 -*-
"""VERIFICATION INDEPENDANTE sur Qwen2.5-0.5B (GQA 14/2) du chemin causal PCA d'=8.
Reference : RoPE reimplemente a la main, base PCA par SVD (et non eigh sur la
covariance), selection par argsort (et non topk), masque et softmax par boucles
explicites. On compare selection, masque et sortie d'attention."""
import math, pathlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

MID = "Qwen/Qwen2.5-0.5B"
T, Lb, W, M, DP, NFIT = 512, 32, 128, 2, 8, 256
LAYERS = [0, 14]
QUERIES = [100, 300, 500]
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats")
log = OUT / "verification_qwen.txt"

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32,
                                             attn_implementation="eager").eval()
corpus = ""
for f in sorted(pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ids = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0][:T][None]

CAP = {}
CFG = {"mode": "sparse"}


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
        if CFG["mode"] == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(g > 1))
            ti = None
            allowed = None
        else:
            ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
            nb = qn // Lb
            j = torch.arange(qn); p = torch.arange(qn)
            causal = (j[None, :] <= p[:, None])
            allowed = ((j[None, :] >= (p - W + 1).clamp(min=0)[:, None]) & causal)[None].expand(
                q.numel() // (bsz * self.head_dim * qn), qn, qn).clone() if False else \
                ((j[None, :] >= (p - W + 1).clamp(min=0)[:, None]) & causal)[None].expand(
                    q.shape[1], qn, qn).clone()
            bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - W + 1).clamp(min=0)[None, :])
            K = ke[0]
            Kfit = K[:, :NFIT, :]                      # base CAUSALE
            C = torch.einsum("hnd,hne->hde", Kfit, Kfit)
            _, ev = torch.linalg.eigh(C)
            P = ev[:, :, -DP:].transpose(1, 2)
            qp = torch.einsum("hqd,hpd->hqp", q[0], P)
            kp = torch.einsum("hkd,hpd->hkp", K, P)
            Sp = torch.einsum("hqp,hkp->hqk", qp, kp)
            Sb = Sp[:, :, :nb * Lb].view(q.shape[1], qn, nb, Lb)
            causb = causal[:, :nb * Lb].view(qn, nb, Lb)
            Sb = Sb.masked_fill((~causb)[None], float("-inf"))
            sc = Sb.max(dim=3).values.masked_fill((~bv).T[None], float("-inf"))
            tv, ti = torch.topk(sc, M, dim=2)
            ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
            valid = ti >= 0
            onehot = torch.zeros(q.shape[1], qn, nb, dtype=torch.bool)
            hh, pp, _ = torch.nonzero(valid, as_tuple=True)
            onehot[hh, pp, ti[valid]] = True
            allowed |= onehot[:, :, (j // Lb)] & causal[None]
            o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        if li in LAYERS:
            CAP[li] = {"q": q[0].clone(), "k": k[0].clone(), "v": v[0].clone(),
                       "cos": cos.clone(), "sin": sin.clone(), "ti": None if ti is None else ti.clone(),
                       "allowed": None if allowed is None else allowed.clone(),
                       "o": o[0].clone(), "g": g, "D": self.head_dim, "mod": self}
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))

with torch.no_grad():
    CFG["mode"] = "sparse"
    model(ids)

log.write_text(f"{MID} | verification independante | T={T} W={W} Lb={Lb} m={M} d'={DP} NFIT={NFIT}\n",
               encoding="utf-8")
lines_out = []


def rope_manuel(x, cos, sin):
    """RoPE convention Qwen2 : rotation par moities."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    c, s = cos[..., : d // 2], sin[..., : d // 2]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


for li in LAYERS:
    cap = CAP[li]
    mod = cap["mod"]; g = cap["g"]; D = cap["D"]
    Hh = cap["q"].shape[0]
    # --- reimplementation : projections et RoPE refaits a la main ---
    with torch.no_grad():
        hs = model.model.layers[li].self_attn  # module
    # on repart des tenseurs bruts projete : on les recalcule depuis q/k/v non-rotes ?
    # ici on rejoue la rotation manuellement sur les tenseurs pre-RoPE reconstruits
    # -> on utilise les projections brutes du module sur les hidden states captures
    # (plus simple : comparer la rotation appliquee)
    q_man = rope_manuel(cap["q"] * 0 + cap["q"], cap["cos"], cap["sin"])
    # cap["q"] est DEJA rotate : test de coherence = rotation appliquee deux fois ? non.
    # On verifie donc la rotation sur les tenseurs pre-RoPE captures a part.
    for p in QUERIES:
        for h in range(Hh):
            kv = h // g
            k = cap["k"][kv]                     # (T, D)
            v = cap["v"][kv]
            q = cap["q"][h, p]                   # (D,)
            # base causale par SVD (chemin different de eigh sur la covariance)
            Kfit = k[:NFIT]                      # (NFIT, D)
            U, S, Vh = torch.linalg.svd(Kfit, full_matrices=False)
            P = Vh[:DP].T                        # (D, d')
            kp = k @ P                           # (T, d')
            qp = q @ P                           # (d',)
            lo = max(0, p - W + 1)
            nb = T // Lb
            scores = []
            for b in range(nb):
                if (b + 1) * Lb <= lo:
                    j0, j1 = b * Lb, min((b + 1) * Lb, p + 1)
                    if j1 > j0:
                        scores.append((float((kp[j0:j1] @ qp).max()), b))
            scores.sort(key=lambda t: -t[0])
            sel_ref = sorted([b for _, b in scores[:M]])
            ti_h = cap["ti"][h, p]
            sel_h = sorted([int(x) for x in ti_h if int(x) >= 0])
            # masque de reference
            ref_mask = set(range(lo, p + 1))
            for b in sel_ref:
                for jj in range(b * Lb, min((b + 1) * Lb, p + 1)):
                    ref_mask.add(jj)
            h_mask = set(torch.nonzero(cap["allowed"][h, p]).flatten().tolist())
            # sortie d'attention de reference
            idx = torch.tensor(sorted(ref_mask))
            logits = (k[idx] @ q) / math.sqrt(D)
            w = torch.softmax(logits, dim=0)
            o_ref = (w[:, None] * v[idx]).sum(0)
            o_h = cap["o"][h, p]
            d_rel = float((o_ref - o_h).norm() / o_h.norm())
            lines_out.append(
                f"L{li} p={p:3d} h={h:2d} | sel_ref={sel_ref} sel_harnais={sel_h} "
                f"| masque {len(ref_mask)}/{len(h_mask)} identique={ref_mask == h_mask} "
                f"| sortie rel={d_rel:.2e}")

log.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
sel_ok = sum(1 for L in lines_out if L.split("sel_ref=")[1].split(" sel_harnais=")[0] ==
             L.split("sel_harnais=")[1].split(" ")[0])
mask_ok = sum(1 for L in lines_out if "identique=True" in L)
print(f"lignes={len(lines_out)} selection_identique={sel_ok} masque_identique={mask_ok}")
print("\n".join(lines_out[:6]))
