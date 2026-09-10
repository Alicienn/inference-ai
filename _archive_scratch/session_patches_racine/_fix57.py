# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\gpt2_source_bloc.py")
s = p.read_text(encoding="utf-8")
old = ("            pk = A.argmax(-1)\n"
       "            ih = torch.arange(h).unsqueeze(1)\n"
       "            iq = torch.arange(qn).unsqueeze(0)\n"
       "            keep = mask[ih, iq, pk]\n")
new = ("            pk = A.argmax(-1)\n"
       "            pkb = pk // LB\n"
       "            hit = (sel == pkb[..., None]).any(-1)\n"
       "            inwin = pk > (torch.arange(qn)[None, :] - W)\n"
       "            keep = hit | inwin\n")
assert old in s, "bloc introuvable"
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("patch ok")
