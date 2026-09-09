# -*- coding: utf-8 -*-
"""Integrite des captures : la sequence est-elle contigue, ou tuilee tous les 64 tokens ?
Test 1 : similarite cosinus entre K[j] et K[j+lag].
Test 2 : meta du npz (tokens stockes ?)."""
import numpy as np, json, pathlib
P = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\12_poc\resultats")
for tag in ("smol8k", "qwen8k"):
    d = np.load(P / f"qk_{tag}.npz", mmap_mode="r")
    meta = json.loads(str(d["meta"]))
    print(f"\n=== {tag} ===")
    print("  cles meta :", list(meta.keys()))
    for k in ("model", "dataset", "seq_len", "T", "text", "tokens", "n_docs"):
        if k in meta:
            v = meta[k]
            print(f"    {k} = {str(v)[:120]}")
    layer = meta["layers"][0]
    K = np.asarray(d[f"0_{layer}_k_rope"])[:, 0, :].astype(np.float64)
    print("  cosinus moyen entre K[j] et K[j+lag] :")
    for lag in (1, 2, 8, 32, 63, 64, 65, 128, 192):
        a, b = K[:-lag], K[lag:]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        print(f"    lag={lag:4d}  cos={cos.mean():+.4f}  (min {cos.min():+.3f}, max {cos.max():+.3f})")
