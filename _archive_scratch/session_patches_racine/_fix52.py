# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
src = (root / "20_sortie_attention" / "RAPPORT_v24.md").read_text(encoding="utf-8")
add = """
## Le gain croît quand on lit moins

Balayage de la fraction lue à T=1024 (fenêtre et quota mis à l'échelle ensemble, Lb=4, index déployable, agrégation max) :

| fraction | W | m | clés | perte | Δ vs dense |
|---|---|---|---|---|---|
| 100 % (dense) | — | — | 1024 | 13,5125 | — |
| 37,5 % | 128 | 64 | 384 | 13,4768 | **−0,036** |
| 24,7 % | 85 | 42 | 253 | 13,4359 | **−0,077** |
| 12,3 % | 42 | 21 | 126 | 13,4220 | **−0,090** |
| **6,0 %** | **21** | **10** | **61** | **13,3684** | **−0,144** |

**Monotone** : plus on lit peu, meilleure est la perte.

## Ce que ce banc ne peut pas tester

| sonde | masse dense sur le contenu distant | niveau uniforme | induction active ? |
|---|---|---|---|
| aiguille de 16 tokens placée deux fois | 0,0395 | 0,039 (20/512) | non (perte 14,4 nats) |
| span naturel de 64 tokens répété 4× | 0,263 | 0,250 (64/256) | **oui** (gain +0,395 nat) |

Dans les deux cas, **le contenu distant ne porte aucune attention au-dessus du hasard**, même quand l'induction est mesurablement active. Le taux de sélection du bloc de l'aiguille (75,8 % à 37,5 % de clés lues ; 33,8 % à 12,5 %) est proche ou **inférieur** au taux attendu par sélection aléatoire (78 % et 60 %).

**Conséquence double** : le gain croissant quand la fraction lue diminue est cohérent avec la structure sink + fond plat (rien de concentré à perdre), mais ce banc **ne peut pas** établir que la sparsité préserve la récupération.
"""
src = src.replace("## La fenêtre est un leurre", add + "\n## La fenêtre est un leurre")
src = src.replace("*Version 24 —", "*Version 25 —")
src = src.replace("`index_vs_oracle.py`", "`index_vs_oracle.py`, `fraction_lue.py`, `retention_aiguille.py`, `induction_span.py`")
out = root / "20_sortie_attention" / "RAPPORT_v25.md"
out.write_text(src, encoding="utf-8")
print("ecrit :", out.name, len(src), "octets")
