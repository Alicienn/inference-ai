# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
src = (root / "20_sortie_attention" / "RAPPORT_v21.md").read_text(encoding="utf-8")
add = """
## Ce que le contexte long change

La décomposition répétée sur la grille (T, W) révèle une troisième composante :

| T | W | masse hors fenêtre | support effectif | top-1 | argmax dans les 10 1res | profil des 10 déciles |
|---|---|---|---|---|---|---|
| 512 | 128 | 0,552 | 15,4 | 0,675 | **0,966** | 0,714 / 0,034 / 0,044 / … / 0,035 |
| 1024 | 128 | 0,563 | 30,1 | 0,655 | **0,965** | 0,716 / 0,056 / 0,031 / … / 0,041 |
| 2048 | 128 | 0,588 | 59,4 | 0,568 | **0,917** | 0,633 / 0,041 / 0,026 / 0,018 / 0,019 / 0,034 / 0,040 / 0,047 / 0,060 / **0,082** |
| 2048 | 256 | 0,533 | 52,9 | 0,608 | **0,917** | 0,676 / 0,044 / 0,028 / … / **0,060** |

Le sink tient à toutes les longueurs — 92 à 97 % des argmax restent dans les dix premières positions — mais **sa part baisse** (0,71 → 0,63) et, à T=2048, **le fond n'est plus plat** : minimum aux déciles 3–4 (0,018) puis **remontée monotone jusqu'à 0,082 au dernier décile**, c'est-à-dire juste derrière la fenêtre. C'est un **gradient de récence**, et il place le trou de bordure dans la zone **la plus dense** du fond — le pire endroit pour perdre des clés.

La pénalité mesurée reste néanmoins faible (0,004 nat à T=2048 en configuration mise à l'échelle), donc le gradient **déplace le mécanisme sans l'inverser**. Le support effectif double quand T double (15 → 30 → 59 clés).
"""
src = src.replace("## La fenêtre est un leurre", add + "\n## La fenêtre est un leurre")
src = src.replace("*Version 21 —", "*Version 22 —")
src = src.replace("`sink_verif.py`", "`sink_verif.py`, `grille_sink.py`")
src = src.replace("structure vérifiée sur trois architectures", "structure vérifiée sur trois architectures et sur la grille (T,W)")
out = root / "20_sortie_attention" / "RAPPORT_v22.md"
out.write_text(src, encoding="utf-8")
print("ecrit :", out, len(src), "octets")
