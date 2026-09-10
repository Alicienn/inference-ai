import pathlib
src = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\courbe_octets.py")
t = src.read_text(encoding="utf-8")
t = t.replace('MID = "HuggingFaceTB/SmolLM2-135M"', 'MID = "Qwen/Qwen2.5-0.5B"')
t = t.replace("DPS = [4, 8, 16, 32]", "DPS = [8, 16]")
t = t.replace("from transformers.models.llama.modeling_llama import apply_rotary_pos_emb",
              "from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb")
t = t.replace('"courbe_octets.txt"', '"courbe_octets_qwen.txt"')
out = src.with_name("courbe_octets_qwen.py")
out.write_text(t, encoding="utf-8")
print("script Qwen genere")
