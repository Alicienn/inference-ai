import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\err_attention.py")
t = p.read_text(encoding="utf-8")
t = t.replace("            keep[i, b * Lb:(b + 1) * Lb] = True",
              "            keep[i, int(b) * Lb:(int(b) + 1) * Lb] = True")
t = t.replace("            keep1[i, b * Lb:(b + 1) * Lb] = True",
              "            keep1[i, int(b) * Lb:(int(b) + 1) * Lb] = True")
p.write_text(t, encoding="utf-8")
print("patche:", t.count("int(b) * Lb"), "occurrences")
