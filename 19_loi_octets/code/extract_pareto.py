import json, pathlib
RES = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\resultats")
d = json.loads((RES / "asp_pareto.json").read_text(encoding="utf-8"))
lbl = "qwen8k|RoPE"
res = d[lbl]
asp, flat = res["asp"], res["flat"]
print("n_blocs =", res["n_blocs_moyen"])
print("\n--- toutes les configurations a deux passes, triees par octets ---")
print(f"{'config':44s} {'octets':>9s} {'rappel':>8s}")
for k, v in sorted(asp.items(), key=lambda kv: kv[1]["octets"]):
    print(f"{k:44s} {v['octets']:9.0f} {100*v['rappel']:7.2f}%")
print("\n--- comparaison A BUDGET EGAL : passe2 coreset vs passe2 quantifiee ---")
# pour chaque (p1, m1), trouver le meilleur coreset et la meilleure quant a budget comparable
import collections
groups = collections.defaultdict(dict)
for k, v in asp.items():
    p1, rest = k.split(" -> ")
    f2, m1 = rest.split(" ")
    groups[(p1, m1)][f2] = v
for (p1, m1), g in sorted(groups.items()):
    cs = [(v["octets"], v["rappel"], f) for f, v in g.items() if f.startswith("coreset")]
    qt = [(v["octets"], v["rappel"], f) for f, v in g.items() if f.startswith("quant")]
    ex = [(v["octets"], v["rappel"], f) for f, v in g.items() if f == "exact"]
    print(f"  {p1} -> [passe2]  {m1}")
    for tag, lst in (("coreset", cs), ("quant  ", qt), ("exact  ", ex)):
        if lst:
            b, r, f = max(lst, key=lambda t: t[1])
            print(f"      {tag} meilleur : {f:14s} {b:9.0f} o  {100*r:6.2f}%")
    # le point cle : a octets egaux (~48k), que donnent les deux familles ?
print("\n--- a budget MATCHED (~48 ko) ---")
for k, v in sorted(asp.items(), key=lambda kv: kv[1]["octets"]):
    if 40000 <= v["octets"] <= 60000:
        print(f"  {k:44s} {v['octets']:9.0f} o  {100*v['rappel']:6.2f}%")
print("\n--- plats ---")
for k, v in sorted(flat.items(), key=lambda kv: kv[1]["octets"]):
    print(f"  {k:20s} {v['octets']:9.0f} o  {100*v['rappel']:6.2f}%")
