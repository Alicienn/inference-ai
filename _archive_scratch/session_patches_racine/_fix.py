import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\masse_vs_utilite.py")
t = p.read_text(encoding="utf-8")
old = 'o = torch.einsum("hqk,hkd->hqd", A, ve)'
new = 'o = torch.einsum("hqk,hkd->hqd", A, ve[0])'
assert old in t, "motif absent"
p.write_text(t.replace(old, new), encoding="utf-8")
print("corrige")
