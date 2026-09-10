import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\index_appris.py")
t = p.read_text(encoding="utf-8")
t = t.replace("def make_fwd():", "def make_fwd(li):")
t = t.replace('                    P = CFG["P"]\n', '                    P = CFG["P"][li]\n')
t = t.replace("for layer in model.model.layers:\n    layer.self_attn.forward = make_fwd().__get__(layer.self_attn, type(layer.self_attn))",
              "for li, layer in enumerate(model.model.layers):\n    layer.self_attn.forward = make_fwd(li).__get__(layer.self_attn, type(layer.self_attn))")
# evaluation supplementaire sur le texte d'entrainement B (detection de sur-apprentissage)
old = 'print("ok")'
new = '''
with log.open("a", encoding="utf-8") as f:
    f.write("\\n=== sur-apprentissage : texte B (entrainement) vs texte A (tenu a l'ecart) ===\\n")
    f.flush()
resB = {}
for dp in (4, 8):
    for tag in ("pcaB", "learned"):
        CFG.update({"sel": "proj", "P": BASES[(tag, dp)]})
        v = float(loss(IDS_B).mean())
        resB[f"{tag}_d{dp}"] = v
        with log.open("a", encoding="utf-8") as f:
            f.write(f"[B] {tag}_d{dp} perte={v:.4f}  |  [A] {res[f'{tag}_d{dp}']:.4f}  "
                    f"ecart B-A={v - res[f'{tag}_d{dp}']:+.4f}\\n")
            f.flush()
print("ok")'''
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("corrige : indexation par couche + test de sur-apprentissage")
