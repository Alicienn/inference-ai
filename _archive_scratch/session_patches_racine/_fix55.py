# -*- coding: utf-8 -*-
import pathlib, re
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\gpt2_source_bloc.py")
s = p.read_text(encoding="utf-8")
s = s.replace('ACC = {"sel": 0, "n": 0, "nl": 0, "m1": 0.0}',
              'ACC = {"sel": 0, "n": 0, "nl": 0, "m1": 0.0, "peak": 0, "peakn": 0}')
ins = ('            pk = A.argmax(-1)\n'
       '            keep = mask.gather(2, pk.unsqueeze(-1)).squeeze(-1)\n'
       '            ACC["peak"] += int(keep[:, QPOS].sum())\n'
       '            ACC["peakn"] += keep[:, QPOS].numel()\n'
       '            A = A * mask\n')
assert "            A = A * mask\n" in s
s = s.replace("            A = A * mask\n", ins, 1)
s = s.replace('"config        perte_copie4  Delta_vs_dense  bloc_source_sel%\\n"',
              '"config        perte_copie4  Delta_vs_dense  bloc_source_sel%  bloc_du_PIC%\\n"')
s = s.replace('selp = 100.0 * ACC["sel"] / max(ACC["n"], 1) if mode != "dense" else float("nan")',
              'selp = 100.0 * ACC["sel"] / max(ACC["n"], 1) if mode != "dense" else float("nan")\n'
              '    peakp = 100.0 * ACC["peak"] / max(ACC["peakn"], 1) if mode != "dense" else float("nan")')
s = s.replace('f.write(f"{tag:12s} {l4:12.5f}  {l4 - D:+.5f}      {selp:6.1f}\\n")',
              'f.write(f"{tag:12s} {l4:12.5f}  {l4 - D:+.5f}      {selp:6.1f}         {peakp:6.1f}\\n")')
s = s.replace('print(f"{tag}: perte_copie4={l4:.5f} (Delta {l4-D:+.5f}) bloc_source_sel={selp:.1f}%", flush=True)',
              'print(f"{tag}: perte_copie4={l4:.5f} (Delta {l4-D:+.5f}) bloc_source_sel={selp:.1f}% bloc_du_PIC={peakp:.1f}%", flush=True)')
s = s.replace('log = OUT / "gpt2_source_bloc.txt"', 'log = OUT / "gpt2_pic_bloc.txt"')
p.write_text(s, encoding="utf-8")
print("patch ok")
