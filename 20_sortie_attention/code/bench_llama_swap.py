# -*- coding: utf-8 -*-
"""Portage du banc a deux etages sur un modele Llama (SmolLM2-135M / Qwen2.5-0.5B).
Hooks : capture Q/K post-RoPE, remplacement de la sortie d'attention par la sortie creuse.
Banc d'induction : span de 64 tokens repete 4x, perte mesuree sur les positions 193-255."""
import math, pathlib, os, torch, torch.nn.functional as F, random
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = os.environ.get("BENCH_MODEL", "HuggingFaceTB/SmolLM2-135M")
TAG = os.environ.get("BENCH_TAG", "smol135")
T, NBASE, LB, NT, SPAN, DMAX = 256, 128, 4, 4, 64, 64
QPOS = list(range(193, 256))
CFG = {"mode": "dense", "W": 32, "m": 16, "dp": 8, "kagg": "max", "bagg": "max", "two": "none"}
BASE = {}
CAP = {}
ACC = {"peak": 0, "peakn": 0, "S": 0.0, "L1": 0.0, "nl": 0}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, attn_implementation="eager").eval()
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
ALL = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[0]
print(f"modele={MID} | vocab ok | corpus {len(ALL)} tokens", flush=True)


def rot(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)


def postrope(li, qn):
    att = model.model.layers[li].self_attn
    cfg = att.config
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nh) or nh
    hd = getattr(att, "head_dim", None) or cfg.hidden_size // nh
    q = CAP[(li, "q")].view(1, qn, nh, hd).transpose(1, 2)
    k = CAP[(li, "k")].view(1, qn, nkv, hd).transpose(1, 2)
    v = CAP[(li, "v")].view(1, qn, nkv, hd).transpose(1, 2)
    cos, sin = CAP[("rot", "cos")], CAP[("rot", "sin")]
    if cos.dim() == 2:
        cos, sin = cos[None], sin[None]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q = q * cos + rot(q) * sin
    k = k * cos + rot(k) * sin
    rep = nh // nkv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return q[0], k[0], v[0], att


def att_hook(li):
    def h(module, args, out):
        qn = CAP[(li, "q")].shape[1]
        q, k, v, att = postrope(li, qn)
        h_, _, hd = q.shape
        j = torch.arange(qn)
        S = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)
        S = S.masked_fill(j[None, None, :] > j[None, :, None], float("-inf"))
        A3 = torch.softmax(S, dim=-1)
        del S
        if CFG["mode"] == "collect":
            K = k.float()
            Kc = K - K.mean(1, keepdim=True)
            U, _, _ = torch.linalg.svd(Kc.transpose(1, 2), full_matrices=False)
            BASE[li] = U[:, :, :DMAX].contiguous()
            return out
        if CFG["mode"] != "pca" or CFG["m"] <= 0 or qn != T:
            return out
        W, m = CFG["W"], CFG["m"]
        nb = qn // LB
        U = BASE[li][:, :, :CFG["dp"]].to(q.dtype)
        qp = torch.einsum("hqd,hde->hqe", q, U)
        kp = torch.einsum("hkd,hde->hke", k, U)
        prod = qp[:, :, None, :] * kp[:, None, :, :]
        sc = prod.amax(-1) if CFG["kagg"] == "max" else prod.sum(-1)
        del prod, qp, kp
        if CFG.get("bscore", "pca") == "norm":
            kn = k.norm(dim=-1)[:, :nb * LB].view(h_, nb, LB)
            B = kn.amax(-1)[:, None, :].expand(h_, qn, nb).contiguous()
        else:
            v4 = sc[:, :, :nb * LB].view(h_, qn, nb, LB)
            B = v4.amax(-1) if CFG["bagg"] == "max" else v4.sum(-1)
            del v4
        bid = torch.arange(nb)
        elig = (bid[None, :] * LB + LB - 1) <= (torch.arange(qn) - W)[:, None]
        Bm = B * elig[None]
        two = CFG["two"]
        if two == "none":
            _, sel = torch.topk(Bm, min(m, nb), dim=2)
        else:
            ncan = min(2 * m, nb)
            _, cand = torch.topk(Bm, ncan, dim=2)
            if two == "exactmax":
                Bc = A3[:, :, :nb * LB].view(h_, qn, nb, LB).amax(-1).gather(2, cand)
            elif two == "exactmass":
                Bc = A3[:, :, :nb * LB].view(h_, qn, nb, LB).sum(-1).gather(2, cand)
            elif two in ("q8h", "q4h"):
                lv = float(2 ** (8 if two == "q8h" else 4) - 1)
                scl = k.abs().amax(dim=(1, 2), keepdim=True)
                kq = torch.round(k / scl.clamp_min(1e-8) * lv).clamp(-lv, lv) / lv * scl
                scq = torch.einsum("hqd,hkd->hqk", q, kq)
                del kq
                Bc = scq[:, :, :nb * LB].view(h_, qn, nb, LB).amax(-1).gather(2, cand)
                del scq
            else:
                Bc = torch.rand(h_, qn, ncan)
            _, sel2 = torch.topk(Bc, min(m, ncan), dim=2)
            sel = cand.gather(2, sel2)

            if CFG.get("swapk", 0) > 0:
                kk = CFG["swapk"]
                hh, qq, nn = Bm.shape
                selb = torch.zeros_like(Bm, dtype=torch.bool); selb.scatter_(2, sel, True)
                candb = torch.zeros_like(Bm, dtype=torch.bool); candb.scatter_(2, cand, True)
                avail = candb & ~selb
                pri = torch.rand(hh, qq, nn).masked_fill(~selb, 2.0)
                selb.scatter_(2, pri.argsort(2)[:, :, :kk], False)
                pri2 = torch.rand(hh, qq, nn).masked_fill(~avail, 2.0)
                selb.scatter_(2, pri2.argsort(2)[:, :, :kk], True)
                sel = selb.nonzero()[:, 2].view(hh, qq, m)
        keys = (sel[..., None] * LB + torch.arange(LB)).reshape(h_, qn, -1)
        mask = torch.zeros(h_, qn, qn, dtype=torch.bool)
        mask.scatter_(2, keys, True)
        mask |= (j[None, None, :] > (j[None, :, None] - W))
        pk = A3.argmax(-1)
        pkb = pk // LB
        hit = (sel == pkb[..., None]).any(-1)
        inwin = pk > (torch.arange(qn)[None, :] - W)
        keep = hit | inwin
        ACC["peak"] += int(keep[:, QPOS].sum())
        ACC["peakn"] += keep[:, QPOS].numel()
        A3s = A3 * mask
        Sm = A3s.sum(-1)
        A3n = A3s / Sm.clamp(min=1e-9).unsqueeze(-1)
        ACC["S"] += float(Sm[:, QPOS].mean())
        ACC["L1"] += float((A3n - A3).abs().sum(-1)[:, QPOS].mean())
        ACC["nl"] += 1
        o = torch.matmul(A3n.unsqueeze(0), v)
        o = o.permute(0, 2, 1, 3).reshape(1, qn, h_ * hd)
        o = att.o_proj(o)
        del A3, A3s, A3n, mask, B
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    return h


for li, layer in enumerate(model.model.layers):
    att = layer.self_attn
    att.q_proj.register_forward_hook(lambda m, i, o, li=li: CAP.__setitem__((li, "q"), o.detach()))
    att.k_proj.register_forward_hook(lambda m, i, o, li=li: CAP.__setitem__((li, "k"), o.detach()))
    att.v_proj.register_forward_hook(lambda m, i, o, li=li: CAP.__setitem__((li, "v"), o.detach()))

    att.register_forward_hook(att_hook(li))


def rh(m, i, o):
    if isinstance(o, tuple) and len(o) >= 2:
        CAP[("rot", "cos")] = o[0].detach()
        CAP[("rot", "sin")] = o[1].detach()


model.model.rotary_emb.register_forward_hook(rh)

CFG["mode"] = "collect"
with torch.no_grad():
    model(ALL[:NBASE][None])
print("base collectee |", len(BASE), "couches | cos capte:", ("cos" in str(CAP.keys())), flush=True)

seqs = []
rng = random.Random(11)
for i in range(NT):
    s = rng.randint(2000, len(ALL) - SPAN - 8)
    seqs.append(ALL[s:s + SPAN].clone().repeat(4))

log = OUT / f"bench_{TAG}_swap.txt"
log.write_text(f"{MID} | banc induction | index a deux etages | S = masse retenue, L1 = |sparse-dense|\n"
               "cfg        perte      Delta     pic%   S       L1\n", encoding="utf-8")
rows = []
D = 0.0


def run(dp, kagg, bagg, m=16, W=32, two="none", bscore="pca", swapk=0):
    global D
    if dp is None:
        CFG["mode"] = "dense"; CFG["W"] = T // 8; CFG["m"] = 0
    else:
        CFG["mode"] = "pca"; CFG["dp"] = dp; CFG["kagg"] = kagg; CFG["bagg"] = bagg
        CFG["W"] = W; CFG["m"] = m; CFG["two"] = two; CFG["bscore"] = bscore; CFG["swapk"] = swapk; CFG["lab"] = two + (f"-k{swapk}" if swapk else "")
        for kk in ACC:
            ACC[kk] = 0
    l4 = 0.0
    for seq in seqs:
        with torch.no_grad():
            logits = model(seq[None]).logits[0]
        ls = F.cross_entropy(logits[:-1], seq[1:], reduction="none")
        l4 += float(ls[193:T - 1].mean())
    l4 /= NT
    if dp is None:
        with log.open("a", encoding="utf-8") as f:
            f.write(f"---  dense  ---  {l4:9.5f}  ---       ---     ---     ---\n")
        print(f"dense: {l4:.5f}", flush=True)
        return l4
    nl = max(ACC["nl"], 1)
    pk = 100.0 * ACC["peak"] / max(ACC["peakn"], 1)
    Sm, L1 = ACC["S"] / nl, ACC["L1"] / nl
    lab = bscore if bscore != "pca" else CFG.get("lab", two)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{lab:9s}  {l4:9.5f}  {l4 - D:+.5f}  {pk:5.1f}  {Sm:.4f}  {L1:.4f}\n")
    print(f"{lab:9s}: perte={l4:.5f} Delta={l4 - D:+.5f} pic={pk:.1f}% S={Sm:.4f} L1={L1:.4f}", flush=True)
    rows.append((lab, l4 - D, pk, Sm, L1))
    return l4


D = run(None, None, None)
print(f"### frac lue = {(32 + 16 * 4) / 256:.3f} (m=16, W=32)", flush=True)
run(8, "max", "max", m=16, W=32, two="none")
run(8, "max", "max", m=16, W=32, two="exactmax")
run(8, "max", "max", m=16, W=32, two="exactmass")
run(8, "max", "max", m=16, W=32, two="q8h")
run(8, "max", "max", m=16, W=32, two="none", bscore="norm")
print("FIN")
print("### sensibilite SmolLM2 : k remplacements (m=16)", flush=True)
for kk in (0, 1, 2, 4):
    run(8, "max", "max", m=16, W=32, two="exactmax", swapk=kk)
