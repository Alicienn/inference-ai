"""
Capture des Q/K reels (pre- et post-RoPE) d'un modele moderne sur texte long.
Sert de banc d'essai pour l'etude de la selection de blocs.
"""
import numpy as np, torch, pathlib, argparse, json
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
OUT = pathlib.Path(__file__).resolve().parents[1] / "resultats"
OUT.mkdir(exist_ok=True)


def main(mid, seq, ndoc, tag):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=dtype, device_map=device, attn_implementation="sdpa").eval()
    cfg = model.config
    H, KVH = cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", H := cfg.num_attention_heads)
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    L = cfg.num_hidden_layers
    print(f"{mid}: {L} couches, {H} tetes Q, {KVH} tetes KV, head_dim {D}, "
          f"rope_theta {getattr(cfg,'rope_theta','?')}, max_pos {cfg.max_position_embeddings}")

    store = {}
    hooks = []

    def mk(name):
        def fn(mod, inp, out):
            store[name] = out.detach()[0].float().cpu()      # (T, ...)
        return fn

    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(mk(f"q{li}")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(mk(f"k{li}")))

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    buf, texts = "", ""
    allt = []
    for r in ds:
        buf += r["text"]
        if len(buf) > seq * 6:
            allt.append(buf); buf = ""
            if len(allt) >= ndoc: break

    # module rotatif du modele -> cos/sin exacts (meme forme sur toute la famille Qwen)
    model_type = getattr(cfg, "model_type", "qwen2")
    try:
        mod = __import__(f"transformers.models.{model_type}.modeling_{model_type}",
                          fromlist=["apply_rotary_pos_emb"])
        apply_rotary_pos_emb = mod.apply_rotary_pos_emb
    except (ImportError, AttributeError):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    packs = []
    for di, t in enumerate(allt):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seq).input_ids.to(device)
        if ids.shape[1] < seq:
            continue
        emb = model.model.embed_tokens(ids)
        pos = torch.arange(seq, device=device)[None, :]
        cos, sin = model.model.rotary_emb(emb, pos)
        cos, sin = cos.cpu(), sin.cpu()
        model(ids)
        T = seq
        for li in range(L):
            qp = store[f"q{li}"].view(T, H, D)
            kp = store[f"k{li}"].view(T, KVH, D)
            qr, kr = apply_rotary_pos_emb(qp.permute(1, 0, 2)[None],
                                          kp.permute(1, 0, 2)[None], cos, sin)
            packs.append(dict(doc=di, layer=li,
                              q_pre=qp.numpy().astype(np.float16),
                              k_pre=kp.numpy().astype(np.float16),
                              q_rope=qr[0].permute(1, 0, 2).numpy().astype(np.float16),
                              k_rope=kr[0].permute(1, 0, 2).numpy().astype(np.float16)))
        print(f"  doc {di}: {L} couches capturees")

    for h in hooks:
        h.remove()

    # on ne garde qu'un sous-ensemble de couches pour la taille du fichier
    keep = list(range(2, L, max(1, L // 8)))
    sel = [p for p in packs if p["layer"] in keep]
    np.savez_compressed(OUT / f"qk_{tag}.npz",
                        meta=json.dumps(dict(model=mid, seq=seq, H=H, KVH=KVH, D=D,
                                             layers=keep, ndoc=len(allt))),
                        **{f"{p['doc']}_{p['layer']}_{k}": p[k]
                           for p in sel for k in ("q_pre", "k_pre", "q_rope", "k_rope")})
    print(f"-> {OUT/f'qk_{tag}.npz'}  ({len(sel)} paquets, couches {keep})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--ndoc", type=int, default=2)
    ap.add_argument("--tag", default="qwen8k")
    a = ap.parse_args()
    main(a.model, a.seq, a.ndoc, a.tag)
