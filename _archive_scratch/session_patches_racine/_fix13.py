# -*- coding: utf-8 -*-
"""Harnais corrige : ajoute l'oracle (masse exacte) et repare l'auto-test causal."""
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\courbe_corrigee.py")
t = p.read_text(encoding="utf-8")
old = """                if sel == "maxip":
                    Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                else:"""
new = """                if sel in ("maxip", "oracle"):
                    Sp = torch.einsum("hqd,hkd->hqk", q[0], ke[0]) / math.sqrt(D)
                else:"""
assert old in t; t = t.replace(old, new)
old2 = """                Sb = Sb.masked_fill((~causb)[None], float("-inf"))
                sc = Sb.max(dim=3).values
                sc = sc.masked_fill((~bv).T[None], float("-inf"))"""
new2 = """                Sb = Sb.masked_fill((~causb)[None], float("-inf"))
                if sel == "oracle":
                    sc = torch.logsumexp(Sb, dim=3)
                else:
                    sc = Sb.max(dim=3).values
                sc = sc.masked_fill((~bv).T[None], float("-inf"))"""
assert old2 in t; t = t.replace(old2, new2)
old3 = """    CFG.update({"mode": "m0"})
    dt = float((loss(ids)[:255] - loss(ids[:, :256])).abs().max())"""
new3 = """    CFG.update({"mode": "sparse", "sel": "maxip"})
    dt = float((loss(ids)[:255] - loss(ids[:, :256])).abs().max())"""
assert old3 in t; t = t.replace(old3, new3)
old4 = """    CFG.update({"sel": "pca", "dp": 8}); res["pca_d8"] = float(loss(ids).mean())"""
new4 = """    CFG.update({"sel": "oracle"}); res["oracle"] = float(loss(ids).mean())
    CFG.update({"sel": "pca", "dp": 8}); res["pca_d8"] = float(loss(ids).mean())"""
assert old4 in t; t = t.replace(old4, new4)
t = t.replace('for k in ["dense", "maxip", "pca_d8", "aleatoire"]',
              'for k in ["dense", "maxip", "oracle", "pca_d8", "aleatoire"]')
t = t.replace('courbe_corrigee.txt', 'courbe_corrigee2.txt')
p.write_text(t, encoding="utf-8")
print("oracle ajoute, auto-test repare")
