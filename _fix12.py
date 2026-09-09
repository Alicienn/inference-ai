import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\courbe_corrigee.py")
t = p.read_text(encoding="utf-8")
t = t.replace("            hh, pp = torch.nonzero(valid, as_tuple=True)",
              "            hh, pp, _ = torch.nonzero(valid, as_tuple=True)")
p.write_text(t, encoding="utf-8")
print("unpack corrige")
