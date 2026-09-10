# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\gpt2_source_bloc.py")
s = p.read_text(encoding="utf-8")
old = "            keep = mask.gather(2, pk.unsqueeze(-1)).squeeze(-1)\n"
new = ("            ih = torch.arange(h).unsqueeze(1)\n"
       "            iq = torch.arange(qn).unsqueeze(0)\n"
       "            keep = mask[ih, iq, pk]\n")
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("patch ok")
