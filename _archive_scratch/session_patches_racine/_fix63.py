# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
src = (root / "20_sortie_attention" / "RAPPORT_v27.md").read_text(encoding="utf-8")
old = """**Dissociation** : 5,3 points d'écart de rétention du pic, 32× de coût. **Δloss ≈ 4 × (1 − taux de pic)**. La cible n'est pas le rappel de masse mais la **rétention du pic**.

**Le budget achète la fidélité ; la dimension ne l'achète pas** :"""
new = """### La rétention du pic est une fausse piste (auto-correction)

L'hypothèse « il faut retenir le pic » a été **testée et réfutée**. Six variantes d'index au même budget (37,5 %) :

| d′ | projection | bloc | Δloss | pic% | **S (masse retenue)** | L1 |
|---|---|---|---|---|---|---|
| 8 | max | max | +0,226 | 94,4 | **0,8564** | 0,2873 |
| 8 | **sum** | max | +1,105 | **95,2** | 0,7888 | 0,4224 |
| 8 | sum | **sum** | +4,581 | 55,1 | 0,5532 | 0,8937 |
| 16 | sum | max | +0,874 | 96,6 | 0,8023 | 0,3954 |
| 32 | sum | max | +0,494 | **98,5** | 0,8302 | 0,3396 |
| 32 | max | max | **+0,094** | 95,1 | **0,8669** | 0,2661 |

**Corrélations avec la perte** : S → Pearson −0,999, **Spearman −1,000** ; L1 → +0,999 / **+1,000** ; taux de rétention du pic → +0,967 mais **Spearman +0,143**, soit **aucun pouvoir prédictif de rang**. La config la 5× pire (sum/max) retient *mieux* le pic (95,2 %) que la meilleure (max/max, 94,4 %).

**Ce que cela établit** : l'objectif est la **masse d'attention retenue S**, et l'agrégation qui la maximise est **max/max** — remplacer le max sur les projections par une somme fait chuter S de 0,857 à 0,789 (perte ×5) ; remplacer le max sur les clés par une somme la fait chuter à 0,553 (perte ×20).

**Le budget achète S, le rang de projection non** :"""
assert old in src, "bloc introuvable"
src = src.replace(old, new)
src = src.replace("*Version 27 —", "*Version 28 —")
src = src.replace("`gpt2_pic_balayage.py`", "`gpt2_pic_balayage.py`, `gpt2_agregation.py`, `gpt2_masse_pic.py`, `gpt2_metrique.py`")
out = root / "20_sortie_attention" / "RAPPORT_v28.md"
out.write_text(src, encoding="utf-8")
print("ecrit :", out.name, len(src), "octets")
