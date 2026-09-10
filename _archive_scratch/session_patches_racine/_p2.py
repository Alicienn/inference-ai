import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\pass1_compare.py")
t = p.read_text(encoding="utf-8")
t = t.replace('for name, (C, cnt) in list(methods.items()) + [("oracle", None)]:',
              'for name, val in list(methods.items()) + [("oracle", None)]:')
t = t.replace('sc = oracle_sc if C is None else lse_scores(C, cnt, qg, D)',
              'sc = oracle_sc if val is None else lse_scores(val[0], val[1], qg, D)')
t = t.replace("def sweep(tag, doc=0, Lb=64, W=512, nq=6, m=8, bits=4):",
              "def sweep(tag, doc=0, Lb=64, W=512, nq=5, m=8, bits=4):")
p.write_text(t, encoding="utf-8")
print("patche")
