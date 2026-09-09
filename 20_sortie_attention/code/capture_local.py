# -*- coding: utf-8 -*-
"""Le sink est-il une propriete du PREMIER TOKEN, ou un artefact d'un unique BOS ?
Captures SmolLM2-135M (cache local), 2048 tokens, 4 variantes :
  orig_wt      : wikitext doc0 avec BOS  -> validation de la replication RoPE vs qk_smol8k.npz
  wt_nobos     : meme texte SANS token special
  wt_multidoc  : 4 documents de 512 tokens concatenes, chacun avec son BOS (4 sinks potentiels)
  corpus_nobos : texte local du corpus, sans token special
Diagnostics : masse par position, masse du bloc 0, selecteur par cle moyenne vs bloc du sink force.
"""
import os, json, pathlib, numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats"
MID = "HuggingFaceTB/SmolLM2-135M"
T, Lb, W, m, nq = 2048, 64, 512, 8, 4
LAYERS = [0, 5, 10, 15, 20, 25, 29]

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval()
cfg = model.config
H = cfg.num_attention_heads
KVH = getattr(cfg, "num_key_value_heads", H)
D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
grp = H // KVH

store = {}
for li, layer in enumerate(model.model.layers):
    def mk(name):
        def hook(mod, inp, out):
            store[name] = out[0].detach().float()
        return hook
    layer.self_attn.q_proj.register_forward_hook(mk(f"q{li}"))
    layer.self_attn.k_proj.register_forward_hook(mk(f"k{li}"))


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def capture(ids):
    ids = ids[:, :T]
    n = ids.shape[1]
    emb = model.model.embed_tokens(ids)
    pos = torch.arange(n)[None, :]
    cos, sin = model.model.rotary_emb(emb, pos)
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    store.clear()
    model(ids)
    res = {}
    for li in LAYERS:
        qp = store[f"q{li}"].view(n, H, D).transpose(0, 1).unsqueeze(0)
        kp = store[f"k{li}"].view(n, KVH, D).transpose(0, 1).unsqueeze(0)
        qr = (qp * cos + rotate_half(qp) * sin)[0].transpose(0, 1).numpy()
        kr = (kp * cos + rotate_half(kp) * sin)[0].transpose(0, 1).numpy()
        res[li] = (qr.astype(np.float64), kr.astype(np.float64))
    return res


def sma(S):
    S = S - S.max(1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(1, keepdims=True)


def quant_b(Kb, bits=4):
    mn = Kb.min(1, keepdims=True); mx = Kb.max(1, keepdims=True)
    step = np.maximum(mx - mn, 1e-8) / (2 ** bits - 1)
    return np.round((Kb - mn) / step) * step + mn


def diag(cap, tag):
    rows = []
    for li, (Q, K) in cap.items():
        for p in range(T - nq, T):
            for kv in range(KVH):
                qg = Q[p, kv * grp:(kv + 1) * grp]
                Kc = K[:p + 1, kv]
                S = (qg @ Kc.T) / np.sqrt(D)
                P_ = sma(S)
                O = P_ @ Kc
                nO = np.linalg.norm(O, axis=1)
                loc = np.zeros(p + 1, bool); loc[max(0, p - W + 1):p + 1] = True
                nbc = max(0, (p - W + 1) // Lb)
                if nbc < m:
                    continue
                Kb = Kc[:nbc * Lb].reshape(nbc, Lb, D)
                Kq = quant_b(Kb).reshape(nbc * Lb, D)
                sq = (qg @ np.vstack([Kq, Kc[nbc * Lb:]]).T) / np.sqrt(D)
                Sb = S[:, :nbc * Lb].reshape(len(qg), nbc, Lb)
                mx = Sb.max(2, keepdims=True)
                lse = np.log(np.exp(Sb - mx).sum(2)) + mx[:, :, 0]
                sc_mean = (qg @ Kb.mean(1).T) / np.sqrt(D)
                for name, sc, force in (("mean", sc_mean, False), ("mean_sink", sc_mean, True),
                                        ("oracle", lse, False)):
                    if force:
                        order = np.argsort(-sc, 1)
                        tm = np.zeros((len(qg), m), int)
                        for gi in range(len(qg)):
                            tm[gi, 0] = 0
                            tm[gi, 1:] = [b for b in order[gi] if b != 0][:m - 1]
                    else:
                        tm = np.argsort(-sc, 1)[:, :m]
                    kk = np.tile(loc, (len(qg), 1))
                    for gi in range(len(qg)):
                        for b in tm[gi]:
                            kk[gi, int(b) * Lb:(int(b) + 1) * Lb] = True
                    err = np.linalg.norm(sma(np.where(kk, sq, -1e30)) @ Kc - O, axis=1) / nO
                    for gi in range(len(qg)):
                        rows.append({"tag": tag, "layer": li, "p": int(p), "kv": kv, "gi": gi,
                                     "meth": name, "err": float(err[gi]),
                                     "mass0": float(P_[gi, 0]), "massblk0": float(P_[gi, :Lb].sum()),
                                     "massloc": float(P_[gi, loc].sum()),
                                     "masskeep": float((P_[gi] * kk[gi]).sum())})
    return rows


lines = []
# --- variantes d'entree ---
texts = {}
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    buf = ""
    for r in ds:
        buf += r["text"]
        if len(buf) > 8192 * 6:
            texts["wt_doc0"] = buf
            break
    lines.append(f"wikitext doc0 : {len(texts.get('wt_doc0',''))} caracteres")
except Exception as e:
    lines.append(f"wikitext indisponible ({type(e).__name__})")

txt_files = sorted((ROOT / "_extracted").glob("*.txt"))
corpus = ""
for f in txt_files:
    corpus += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
    if len(corpus) > 200000:
        break
texts["corpus"] = corpus
lines.append(f"corpus local : {len(txt_files)} fichiers, {len(corpus)} caracteres")

variants = {}
if "wt_doc0" in texts:
    variants["orig_wt"] = tok(texts["wt_doc0"], return_tensors="pt", truncation=True,
                              max_length=T).input_ids
    variants["wt_nobos"] = tok(texts["wt_doc0"], return_tensors="pt", truncation=True,
                               max_length=T, add_special_tokens=False).input_ids
    chunks = [tok(texts["wt_doc0"][i * 3000:(i + 1) * 3000], return_tensors="pt",
                  truncation=True, max_length=512).input_ids for i in range(4)]
    variants["wt_multidoc"] = torch.cat(chunks, dim=1)[:, :T]
variants["corpus_nobos"] = tok(texts["corpus"], return_tensors="pt", truncation=True,
                               max_length=T, add_special_tokens=False).input_ids

caps = {}
for name, ids in variants.items():
    caps[name] = capture(ids)
    lines.append(f"capture {name} : ids {tuple(ids.shape)} | token[0]={int(ids[0,0])} "
                 f"| tokens uniques aux frontieres={[int(ids[0,i]) for i in (0,512,1024,1536)]}")

# --- validation de la replication RoPE contre le npz existant ---
npz = ROOT / "12_poc" / "resultats" / "qk_smol8k.npz"
if npz.exists() and "orig_wt" in caps:
    d = np.load(npz, mmap_mode="r")
    for li in LAYERS:
        key = f"0_{li}_k_rope"
        if key in d:
            ref = np.asarray(d[key])[:T, 0, :].astype(np.float64)
            mine = caps["orig_wt"][li][1][:, 0, :]
            dd = np.abs(ref - mine).max()
            cs = float(np.sum(ref * mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
            lines.append(f"VALIDATION couche {li} : max|delta K|={dd:.3e}  cosinus={cs:.6f}")
            break
    else:
        lines.append("VALIDATION : aucune couche commune avec le npz")

# --- diagnostics ---
allrows = []
for name, cap in caps.items():
    allrows += diag(cap, name)
    lines.append("")
    lines.append(f"=== {name} ===")
    for meth in ("mean", "mean_sink", "oracle"):
        sel = [x for x in allrows if x["tag"] == name and x["meth"] == meth]
        if not sel:
            continue
        e = np.array([x["err"] for x in sel])
        mk = np.array([x["masskeep"] for x in sel])
        lines.append(f"  {meth:10s} err med={np.median(e):.4f} p90={np.percentile(e,90):.4f} "
                     f"max={e.max():.4f} | masse med={np.median(mk):.3f} "
                     f"| catastrophes={100*(e>0.5).mean():5.1f} %")
    sel = [x for x in allrows if x["tag"] == name]
    m0 = np.array([x["mass0"] for x in sel]); mb = np.array([x["massblk0"] for x in sel])
    ml = np.array([x["massloc"] for x in sel])
    lines.append(f"  masse position 0 = {m0.mean():.4f} (med {np.median(m0):.4f}) | "
                 f"bloc 0 = {mb.mean():.4f} | fenetre 512 = {ml.mean():.4f}")

(OUT / "capture_local_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("ok", len(lines), "lignes")
