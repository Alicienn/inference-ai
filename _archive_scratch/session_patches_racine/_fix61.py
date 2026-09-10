# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\gpt2_metrique.py")
s = p.read_text(encoding="utf-8")
old = "            A3n = A3s / Sm.clamp(min=1e-9)\n"
new = "            A3n = A3s / Sm.clamp(min=1e-9).unsqueeze(-1)\n"
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("patch ok")
