import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\audit_causal_base.py")
t = p.read_text(encoding="utf-8")
t = t.replace('MID = "HuggingFaceTB/SmolLM2-135M"', 'MID = "Qwen/Qwen2.5-0.5B"')
t = t.replace("OFFSETS = [0, 200000, 400000]", "OFFSETS = [0]")
t = t.replace('"audit_causal_base.txt"', '"audit_causal_qwen.txt"')
p.write_text(t, encoding="utf-8")
print("audit Qwen ecrit")
