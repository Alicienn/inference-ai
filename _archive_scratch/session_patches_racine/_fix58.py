# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\gpt2_source_bloc.py")
s = p.read_text(encoding="utf-8")
old = '            keep = hit | inwin\n'
new = ('            keep = hit | inwin\n'
       '            if CFG.get("dbg"):\n'
       '                print("DBG", "h", h, "qn", qn, "m", m, "A", tuple(A.shape),\n'
       '                      "sel", tuple(sel.shape), "pkb", tuple(pkb.shape),\n'
       '                      "hit", tuple(hit.shape), "keep", tuple(keep.shape), flush=True)\n'
       '                CFG["dbg"] = False\n')
assert old in s
s = s.replace(old, new, 1)
s = s.replace('CFG = {"mode": "dense", "W": 32, "m": 16}', 'CFG = {"mode": "dense", "W": 32, "m": 16, "dbg": True}')
p.write_text(s, encoding="utf-8")
print("patch ok")
