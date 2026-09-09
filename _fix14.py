import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\courbe_corrigee.py")
t = p.read_text(encoding="utf-8")
t = t.replace('    CFG.update({"sel": "pca", "dp": 8}); res["pca_d8"] = float(loss(ids).mean())',
              '    CFG.update({"sel": "pca", "dp": 16}); res["pca_d16"] = float(loss(ids).mean())\n'
              '    CFG.update({"sel": "pca", "dp": 8}); res["pca_d8"] = float(loss(ids).mean())')
t = t.replace('for k in ["dense", "maxip", "oracle", "pca_d8", "aleatoire"]',
              'for k in ["dense", "maxip", "oracle", "pca_d16", "pca_d8", "aleatoire"]')
t = t.replace('courbe_corrigee2.txt', 'courbe_corrigee3.txt')
p.write_text(t, encoding="utf-8")
print("pca_d16 ajoute")
