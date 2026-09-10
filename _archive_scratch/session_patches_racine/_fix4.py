import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\base_figee.py")
t = p.read_text(encoding="utf-8")
old = 'CFG["acc"][li] = torch.einsum("hnd,hne->hde", k[0], k[0]).detach()'
new = 'ke0 = k.repeat_interleave(g, 1)[0]\n            CFG["acc"][li] = torch.einsum("hnd,hne->hde", ke0, ke0).detach()'
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("base hors ligne corrigee : tete de requete (H) au lieu des tetes KV")
