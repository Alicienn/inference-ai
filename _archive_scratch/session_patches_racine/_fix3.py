import pathlib, re
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\base_figee.py")
t = p.read_text(encoding="utf-8")
if "def R(k, v):" not in t:
    t = t.replace("res = {}", '''res = {}
def R(k, v):
    res[k] = v
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{k:20s} perte={v:.4f}\\n")
        f.flush()''')
    t = re.sub(r'res\["([^"]+)"\] = (float\(loss\(IDS_A\)\.mean\(\)\))', r'R("\1", \2)', t)
    t = t.replace('res["rand_moy"] = statistics.mean(rands)', 'R("rand_moy", statistics.mean(rands))')
p.write_text(t, encoding="utf-8")
print("journalisation incrementale installee")
