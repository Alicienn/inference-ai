# -*- coding: utf-8 -*-
"""Ratio max|k| / quantile(0,999)|k| par tete (post-RoPE si applicable), multi-architectures."""
import pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
corpus = ""
for f in sorted((ROOT / "_extracted").glob("*.txt")):
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
MODS = ["gpt2", "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B"]


def rot(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)


for MID in MODS:
    try:
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, attn_implementation="eager").eval()
    except Exception as ex:
        print(f"{MID}: INDISPONIBLE ({type(ex).__name__})", flush=True)
        continue
    ids = tok(corpus, return_tensors="pt", add_special_tokens=False).input_ids[:, :256]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers, arch, attname = model.transformer.h, "gpt2", "attn"
    else:
        layers, arch, attname = model.model.layers, "llama", "self_attn"
    nh = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 12))
    nkv = getattr(model.config, "num_key_value_heads", nh) or nh
    cap, hs = {}, []

    def rh(m, i, o):
        if isinstance(o, tuple) and len(o) >= 2:
            cap["cos"], cap["sin"] = o[0].detach(), o[1].detach()
    if arch == "llama" and hasattr(model.model, "rotary_emb"):
        hs.append(model.model.rotary_emb.register_forward_hook(rh))
    for li, layer in enumerate(layers):
        att = getattr(layer, attname)
        nm = "c_attn" if hasattr(att, "c_attn") else "k_proj"
        def hk(m, i, o, li=li):
            cap[li] = o.detach()
        hs.append(getattr(att, nm).register_forward_hook(hk))
    with torch.no_grad():
        model(ids)
    ratios, allr = [], []
    for li in range(len(layers)):
        kk = cap.get(li)
        if kk is None:
            continue
        if arch == "gpt2":
            hd = kk.shape[-1] // (3 * nh)
            kk = kk.view(1, kk.shape[1], 3, nh, hd)[:, :, 1].transpose(1, 2)
        else:
            hd = kk.shape[-1] // nkv
            kk = kk.view(1, kk.shape[1], nkv, hd).transpose(1, 2)
            if "cos" in cap:
                cos, sin = cap["cos"], cap["sin"]
                if cos.dim() == 2:
                    cos, sin = cos[None], sin[None]
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
                kk = kk * cos + rot(kk) * sin
        fl = kk.abs().reshape(kk.shape[1], -1).float()
        q999 = torch.quantile(fl, 0.999, dim=1).clamp_min(1e-8)
        r = (fl.amax(1) / q999)
        ratios.append(float(r.mean()))
        allr.append(r)
    R = torch.tensor(ratios)
    mx = max(float(r.max()) for r in allr)
    print(f"{MID:34s} arch={arch:5s} couches={len(ratios):3d} ratio max/p99.9 "
          f"moyen={R.mean():5.2f} median={R.median():5.2f} max sur les tetes={mx:6.2f}", flush=True)
    for h in hs:
        h.remove()
