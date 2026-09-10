# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
src = (root / "20_sortie_attention" / "RAPPORT_v23.md").read_text(encoding="utf-8")
add = """
## L'index déployable bat le dense

Le contraste ci-dessus est oracle. Avec l'index **causal déployable** (base PCA d′=8 sur les 256 premiers tokens, score de bloc = max sur les clés du max sur 8 projections), à la même fraction lue et Lb=4 :

| T | dense | oracle (Δ) | **index PCA (Δ)** |
|---|---|---|---|
| 1024 | 13,5125 | 13,4733 (**−0,039**) | 13,4768 (**−0,036**) |
| 2048 | 13,5443 | 13,5179 (**−0,026**) | 13,4977 (**−0,047**) |

**Les deux configurations creuses battent le dense**, y compris avec l'index déployable ; l'index récupère 91 % du gain de l'oracle à T=1024 et **bat l'oracle** à T=2048.

**Mais** — fait le plus important — **le signe dépend de l'agrégation du score de bloc**. Avec une agrégation par **somme** de masse, le même point T=1024 donne **+0,022** ; avec un **max**, il donne **−0,039**. L'agrégation est un choix de premier ordre, pas un détail d'implémentation : cela étend au niveau du bloc le principe « le max bat la moyenne » du POC initial.

Réserves : **perte** (NLL) à 2,7 × de réduction KV, pas une latence ; sélection à 37,5 % des clés, pas le régime 13–25 × ; un seul modèle de 135 M ; T=2048 sur une seule tranche.
"""
src = src.replace("## La fenêtre est un leurre", add + "\n## La fenêtre est un leurre")
src = src.replace("*Version 23 —", "*Version 24 —")
src = src.replace("`kappa_T_propre.py`", "`kappa_T_propre.py`, `index_vs_oracle.py`")
out = root / "20_sortie_attention" / "RAPPORT_v24.md"
out.write_text(src, encoding="utf-8")
print("ecrit :", out.name, len(src), "octets")
