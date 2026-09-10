# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
src = (root / "20_sortie_attention" / "RAPPORT_v26.md").read_text(encoding="utf-8")
add = """## Le banc de récupération existe : GPT-2

La dégénérescence n'est pas universelle. Sur **GPT-2**, la même sonde engage un vrai circuit de récupération : la perte sur la 4ᵉ copie est **2,95 nat** sous la 1ʳᵉ, l'attention sur la copie source distante vaut **0,437** contre 0,250 uniforme, et une tête de la couche 11 y met **la totalité** de sa masse.

| config | perte copie 4 | Δ vs dense | bloc source « sélectionné » | **bloc du pic retenu** |
|---|---|---|---|---|
| dense | 0,06211 | — | — | — |
| oracle 37,5 % | 0,06917 | **+0,007** | 98,0 % | **99,7 %** |
| index PCA 37,5 % | 0,28764 | **+0,226** | 98,3 % | **94,4 %** |
| index PCA 12,5 % | 1,46088 | +1,399 | 86,8 % | 74,9 % |

**Dissociation** : 5,3 points d'écart de rétention du pic, 32× de coût. **Δloss ≈ 4 × (1 − taux de pic)**. La cible n'est pas le rappel de masse mais la **rétention du pic**.

**Le budget achète la fidélité ; la dimension ne l'achète pas** :

| d′ | fraction | perte | Δ | pic retenu |
|---|---|---|---|---|
| 8 | 25 % | 0,64276 | +0,581 | 89,6 % |
| 8 | 37,5 % | 0,28764 | +0,226 | 94,4 % |
| 8 | 50 % | 0,13222 | +0,070 | 96,9 % |
| 8 | **75 %** | **0,06078** | **−0,001** | **99,5 %** |
| 16 | 37,5 % | 0,22409 | +0,162 | 94,7 % |
| 32 | 37,5 % | 0,15562 | +0,094 | 95,1 % |

**Verdict** : sur une tâche de récupération ASP économise **~1,3×** (75 % de clés lues), pas les 16× du texte continu — les deux régimes sont aux extrémités opposées de l'axe du budget.

"""
src = src.replace("## La fenêtre est un leurre", add + "## La fenêtre est un leurre")
src = src.replace("*Version 26 —", "*Version 27 —")
src = src.replace("`induction_span_qwen.py`", "`induction_span_qwen.py`, `induction_span_gpt2.py`, `gpt2_pic_bloc.py`, `gpt2_pic_balayage.py`")
out = root / "20_sortie_attention" / "RAPPORT_v27.md"
out.write_text(src, encoding="utf-8")
print("ecrit :", out.name, len(src), "octets")
