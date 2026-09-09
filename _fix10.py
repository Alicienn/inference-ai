import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\verification.py")
t = p.read_text(encoding="utf-8")
old = "            sc = torch.stack([(q0[:, p] * ke[:, b * Lb:(b + 1) * Lb]).sum(-1).max(-1).values\n                              for b in blk], dim=1)"
new = "            sc = torch.stack([(q0[:, p].unsqueeze(1) * ke[:, b * Lb:(b + 1) * Lb]).sum(-1).max(-1).values\n                              for b in blk], dim=1)"
assert old in t
p.write_text(t.replace(old, new), encoding="utf-8")
print("broadcast corrige")
