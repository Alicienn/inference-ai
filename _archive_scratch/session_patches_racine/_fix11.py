import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\courbe_corrigee.py")
t = p.read_text(encoding="utf-8")
old = """            valid = ti >= 0
            onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
            onehot.scatter_(2, torch.where(valid, ti, torch.zeros_like(ti)), True)
            onehot &= valid"""
new = """            valid = ti >= 0
            onehot = torch.zeros(H, qn, nb, dtype=torch.bool)
            hh, pp = torch.nonzero(valid, as_tuple=True)
            onehot[hh, pp, ti[valid]] = True"""
assert old in t
p.write_text(t.replace(old, new), encoding="utf-8")
print("correction sentinelle appliquee")
