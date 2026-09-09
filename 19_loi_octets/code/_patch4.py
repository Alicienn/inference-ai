import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code\frontiere2.py")
t = p.read_text(encoding="utf-8")
old = '''    for k in (2, 3, 4, 6, 8):
        lo = Ks.min(1, keepdims=True); hi = Ks.max(1, keepdims=True)
        rng = np.maximum(hi - lo, 1e-12)
        q = np.round((Ks - lo) / rng * (2 ** k - 1))
        Kq = lo + q / (2 ** k - 1) * rng
        F[f"quant {k}b"] = ("keys", dict(K=Kq, ratio=np.ones(n)), L * D * k / 8.0 + 4)'''
new = '''    for k in (1, 2, 3, 4, 6, 8):
        # (a) echelles GLOBALES par dimension (calculees une fois au prefill, comme
        #     dans les quantifieurs KV de production : cout amorti)
        lo = Ks.min((0, 1), keepdims=True); hi = Ks.max((0, 1), keepdims=True)
        rng = np.maximum(hi - lo, 1e-12)
        q = np.round((Ks - lo) / rng * (2 ** k - 1))
        F[f"quantG {k}b"] = ("keys", dict(K=lo + q / (2 ** k - 1) * rng, ratio=np.ones(n)),
                             L * D * k / 8.0 + 2 * D * 2 / max(n, 1))
        # (b) echelles PAR BLOC et par dimension : +2 scalaires fp16 par dimension
        lo = Ks.min(1, keepdims=True); hi = Ks.max(1, keepdims=True)
        rng = np.maximum(hi - lo, 1e-12)
        q = np.round((Ks - lo) / rng * (2 ** k - 1))
        F[f"quant {k}b"] = ("keys", dict(K=lo + q / (2 ** k - 1) * rng, ratio=np.ones(n)),
                            L * D * k / 8.0 + 2 * D * 2)'''
assert old in t
t = t.replace(old, new)
old2 = '''            for fam, pref in (("coreset", "coreset r="), ("cobs", "cobs r="),
                              ("quant", "quant "), ("sub", "sub r="), ("top", "top r=")):'''
new2 = '''            for fam, pref in (("coreset", "coreset r="), ("cobs", "cobs r="),
                              ("quant", "quant "), ("quantG", "quantG "),
                              ("sub", "sub r="), ("top", "top r=")):'''
assert old2 in t
t = t.replace(old2, new2)
p.write_text(t, encoding="utf-8")
print("patched4 OK")
