import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\verification.py")
t = p.read_text(encoding="utf-8")
t = t.replace('    q = att.q_proj(h_in).view(1, T, H, D).transpose(1, 2)\n    k = att.k_proj(h_in).view(1, T, H, D).transpose(1, 2)\n    v = att.v_proj(h_in).view(1, T, H, D).transpose(1, 2)',
              '    Hkv = att.config.num_key_value_heads if hasattr(att, "config") else model.config.num_key_value_heads\n    q = att.q_proj(h_in).view(1, T, H, D).transpose(1, 2)\n    k = att.k_proj(h_in).view(1, T, Hkv, D).transpose(1, 2)\n    v = att.v_proj(h_in).view(1, T, Hkv, D).transpose(1, 2)')
p.write_text(t, encoding="utf-8")
print("GQA corrige")
