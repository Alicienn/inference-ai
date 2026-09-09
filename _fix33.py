import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\balayage_budget_constant.py")
t = p.read_text(encoding="utf-8")
t = t.replace("CONFIGS = [(64, 1), (32, 2), (16, 4), (8, 8), (4, 16)]",
              "CONFIGS = [(8, 8), (4, 16), (2, 32), (1, 64)]")
t = t.replace('"balayage_budget_constant.txt"', '"balayage_plancher.txt"')
t = t.replace('log.write_text(f"{MID} | T={T} W={W} d\'={DP} | m*Lb=64 constant | index PCA CAUSAL\\n"',
              'log.write_text(f"{MID} | T={T} W={W} d\'={DP} | m*Lb=64 constant | PLANCHER : Lb=1 = selection exacte de 64 cles\\n"')
p.write_text(t, encoding="utf-8")
print("script plancher ecrit")
