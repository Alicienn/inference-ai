import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\audit_causal_base.py")
t = p.read_text(encoding="utf-8")
bad = 'kp = torch.einsum("hkd,hkp->hqk".replace("qk", "kp"), K, P)'
good = 'kp = torch.einsum("hkd,hpd->hkp", K, P)'
assert bad in t
p.write_text(t.replace(bad, good), encoding="utf-8")
print("einsum corrige")
