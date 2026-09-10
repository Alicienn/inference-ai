# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
src = (root / "20_sortie_attention" / "RAPPORT_v25.md").read_text(encoding="utf-8")
old = """| sonde | masse dense sur le contenu distant | niveau uniforme | induction active ? |
|---|---|---|---|
| aiguille de 16 tokens placée deux fois | 0,0395 | 0,039 (20/512) | non (perte 14,4 nats) |
| span naturel de 64 tokens répété 4× | 0,263 | 0,250 (64/256) | **oui** (gain +0,395 nat) |"""
new = """| sonde | masse dense sur le contenu distant | niveau uniforme | induction active ? |
|---|---|---|---|
| SmolLM2-135M, aiguille de 16 tokens placée 2× | 0,0395 | 0,039 (20/512) | non (perte 14,4 nats) |
| SmolLM2-135M, span de 64 tokens répété 4× | 0,263 | 0,250 (64/256) | **oui** (gain +0,395 nat) |
| **Qwen2.5-0.5B**, span de 64 tokens répété 4× | **0,204** (copie proche : 0,287) | 0,250 (64/256) | **non** (gain −0,001 nat) |"""
assert old in src
src = src.replace(old, new)
src = src.replace("Dans les deux cas, **le contenu distant ne porte aucune attention au-dessus du hasard**, même quand l'induction est mesurablement active.",
                  "Dans les trois cas, **le contenu distant ne porte aucune attention au-dessus du hasard**. Chez SmolLM2 l'induction est mesurablement active mais **n'est pas médiée par une attention concentrée** ; chez Qwen2.5-0.5B elle est **absente** et l'attention sur la source distante passe *sous* le niveau uniforme tandis que la copie proche le dépasse — un **tilt de récence**, pas de la récupération. **Passer de 135 M à 0,5 B ne répare pas la limite.**")
src = src.replace("*Version 25 —", "*Version 26 —")
src = src.replace("`induction_span.py`", "`induction_span.py`, `induction_span_qwen.py`")
out = root / "20_sortie_attention" / "RAPPORT_v26.md"
out.write_text(src, encoding="utf-8")
print("ecrit :", out.name, len(src), "octets")
