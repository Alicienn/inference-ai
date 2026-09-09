import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code\frontiere2.py")
t = p.read_text(encoding="utf-8")
old = 'Tm = lse(s * np.einsum("nld,jd->nlj", Ks, Q), 1)'
new = 'Tm = lse(s * np.einsum("nld,jd->nlj", Ks, Q), 1).T'
assert old in t and 'nlj", Ks, Q), 1).T' not in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("patched3 OK")
