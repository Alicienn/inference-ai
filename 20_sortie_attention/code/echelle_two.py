# -*- coding: utf-8 -*-
"""Banc long-contexte instrumente : deuxieme etage (exact / 8 bits) et courbe de sensibilite Delta(k).
Meme banc que echelle_saturee.py (patch du forward Llama, Lb=4, W=T/8, m=T/32 -> 25 % de cles).
Configs via ECHELLE_CFGS : dense, coarse, exact, q8h, rand, swap1, swap2, swap4.
Append dans resultats/echelle_two.txt."""
import math, os, pathlib, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = os.environ.get("ECHELLE_MODEL", "HuggingFaceTB/SmolLM2-135M")
T = int(os.environ.get("ECHELLE_T", "1024"))
W = int(os.environ.get("ECHELLE_W", str(T // 8)))
M = int(os.environ.get("ECHELLE_M", str(T // 32)))
OFF = int(os.environ.get("ECHELLE_OFF", "0"))
DP, NFIT, LB = int(os.environ.get("ECHELLE_DP", "8")), 256, int(os.environ.get("ECHELLE_LB", "4"))
CFGS = os.environ.get("ECHELLE_CFGS", "dense,coarse,exact,q8h").split(",")
CFG = {"mode": "dense", "sel": "pca", "W": W, "Lb": LB, "m": M, "two": "none", "swapk": 0, "nseg": 2}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32, attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
log = OUT / "echelle_two.txt"


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
            Wc, Lb, Mc = CFG["W"], CFG["Lb"], CFG["m"]
            ke = k.repeat_interleave(g, 1); ve = v.repeat_interleave(g, 1)
            h = q.shape[1]; nb = qn // Lb
            j = torch.arange(qn); p = torch.arange(qn)
            causal = (j[None, :] <= p[:, None])
            allowed = ((j[None, :] >= (p - Wc + 1).clamp(min=0)[:, None]) & causal)[None].expand(
                h, qn, qn).clone()
            if Mc > 0:
                bv = ((torch.arange(nb) * Lb + Lb)[:, None] <= (p - Wc + 1).clamp(min=0)[None, :])
                K = ke[0]
                sm = CFG.get("sel", "pca")
                if isinstance(sm, str) and (sm.startswith("sum") or sm.startswith("csum")):
                    Kseg = int(sm[4:]) if sm.startswith("csum") else int(sm[3:])
                    Ks = K[:, :nb * Lb].view(h, nb, Kseg, Lb // Kseg, -1).mean(dim=3).reshape(
                        h, nb * Kseg, -1)
                    if sm.startswith("csum"):
                        Ks = Ks / Ks.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    Sps = torch.einsum("hqd,hkd->hqk", q[0], Ks) / math.sqrt(self.head_dim)
                    sc = Sps[:, :, :nb * Kseg].view(h, qn, nb, Kseg).max(dim=3).values.masked_fill(
                        (~bv).T[None], float("-inf"))
                    del Sps, Ks
                else:
                    C = torch.einsum("hnd,hne->hde", K[:, :NFIT, :], K[:, :NFIT, :])
                    _, ev = torch.linalg.eigh(C)
                    P = ev[:, :, -DP:].transpose(1, 2)
                    qp = torch.einsum("hqd,hpd->hqp", q[0], P)
                    kp = torch.einsum("hkd,hpd->hkp", K, P)
                    Spc = torch.einsum("hqp,hkp->hqk", qp, kp)
                    sc = Spc[:, :, :nb * Lb].view(h, qn, nb, Lb).max(dim=3).values.masked_fill(
                        (~bv).T[None], float("-inf"))
                    del Spc, qp, kp
                two = CFG.get("two", "none")
                if two == "none":
                    tv, ti = torch.topk(sc, Mc, dim=2)
                    ti = torch.where(torch.isfinite(tv), ti, torch.full_like(ti, -1))
                    valid = ti >= 0
                    onehot = torch.zeros(h, qn, nb, dtype=torch.bool)
                    hh, pp, _ = torch.nonzero(valid, as_tuple=True)
                    onehot[hh, pp, ti[valid]] = True
                else:
                    ncan = min(2 * Mc, nb)
                    _, ci = torch.topk(sc, ncan, dim=2)
                    if two == "rand":
                        Bc = torch.rand(h, qn, ncan)
                    else:
                        nseg = Lb
                        if two in ("q1h", "q2h", "q4h", "q8h"):
                            lv = {"q1h": 1.0, "q2h": 3.0, "q4h": 15.0, "q8h": 255.0}[two]
                            scl = K.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
                            Kf = torch.round(K / scl * lv).clamp(-lv, lv) / lv * scl
                        else:
                            Kf = K
                        if two in ("r2", "r1") or two in ("ns", "cns"):
                            nseg = CFG.get("nseg", 2) if two in ("ns", "cns") else (2 if two == "r2" else 1)
                            Kf = Kf[:, :nb * Lb].view(h, nb, nseg, Lb // nseg, -1).mean(dim=3).reshape(
                                h, nb * nseg, -1)
                            if two == "cns":
                                Kf = Kf / Kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        Spf = torch.einsum("hqd,hkd->hqk", q[0], Kf) / math.sqrt(self.head_dim)
                        del Kf
                        Bc = Spf[:, :, :nb * nseg].view(h, qn, nb, nseg).max(dim=3).values.masked_fill(
                            (~bv).T[None], float("-inf")).gather(2, ci)
                        del Spf
                    _, ti2 = torch.topk(Bc, Mc, dim=2)
                    sel = ci.gather(2, ti2)
                    kk = CFG.get("swapk", 0)
                    if kk > 0:
                        selb = torch.zeros(h, qn, nb, dtype=torch.bool); selb.scatter_(2, sel, True)
                        candb = torch.zeros(h, qn, nb, dtype=torch.bool); candb.scatter_(2, ci, True)
                        avail = candb & ~selb
                        pri = torch.rand(h, qn, nb).masked_fill(~selb, 2.0)
                        selb.scatter_(2, pri.argsort(2)[:, :, :kk], False)
                        pri2 = torch.rand(h, qn, nb).masked_fill(~avail, 2.0)
                        selb.scatter_(2, pri2.argsort(2)[:, :, :kk], True)
                        sel = selb.nonzero()[:, 2].view(h, qn, Mc)
                    onehot = torch.zeros(h, qn, nb, dtype=torch.bool)
                    onehot.scatter_(2, sel, True)
                allowed |= onehot[:, :, (j // Lb)] & causal[None]
            o = F.scaled_dot_product_attention(q, ke, ve, attn_mask=allowed)
        return self.o_proj(o.transpose(1, 2).reshape(bsz, qn, -1)), None
    return fwd


for layer in model.model.layers:
    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))


def loss(ids):
    with torch.no_grad():
        out = model(ids)
        return float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:]))


ids = ALL[OFF:OFF + T][None]
with log.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {MID} | T={T} W={W} m={M} offset={OFF} | cfgs={','.join(CFGS)} | "
            f"{100.0 * (W + 2 * M * LB) / T:.1f} % de cles lues ---\n")
    f.write("cfg        perte      vs_dense\n")
CFG.update({"mode": "dense"})
ld = loss(ids)
print(f"dense {ld:.5f} ({time.strftime('%H:%M:%S')})", flush=True)
with log.open("a", encoding="utf-8") as f:
    f.write(f"{'dense':9s}  {ld:9.5f}  ---\n")
for c in CFGS:
    if c == "dense":
        continue
    CFG.update({"mode": "sparse", "sel": "pca", "two": "none", "swapk": 0})
    if c == "coarse":
        pass
    elif c.startswith("ns") and c[2:].isdigit():
        CFG["two"] = "ns"; CFG["nseg"] = int(c[2:])
    elif c.startswith("cns") and c[3:].isdigit():
        CFG["two"] = "cns"; CFG["nseg"] = int(c[3:])
    elif c in ("exact", "q1h", "q2h", "q4h", "q8h", "r2", "r1"):
        CFG["two"] = c
    elif c == "rand":
        CFG["two"] = "rand"
    elif c.startswith("swap"):
        CFG["two"] = "exact"; CFG["swapk"] = int(c[4:])
    elif (c.startswith("sum") and c[3:].isdigit()) or (c.startswith("csum") and c[4:].isdigit()):
        CFG["sel"] = c
    elif c == "maxip":
        CFG["sel"] = "maxip"
    t0 = time.time()
    lp = loss(ids)
    print(f"{c:9s} {lp:.5f}  vs_dense={lp - ld:+.5f}  ({time.time() - t0:.1f}s)", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{c:9s}  {lp:9.5f}  {lp - ld:+.5f}\n")
print("FIN")
