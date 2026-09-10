# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
src = (root / "20_sortie_attention" / "RAPPORT_v22.md").read_text(encoding="utf-8")
add = """
## La constante n'est pas constante

À fraction lue constante (37,5 %, W=T/8, m·Lb=T/4), avec la masse piégée **mesurée** dans le même passage :

| T | dense | Lb=4 (Δ) | M₄ | Lb=16 (Δ) | M₁₆ | κ |
|---|---|---|---|---|---|---|
| 1024 | 13,5125 | 13,5350 (**+0,022**) | 2,52e-3 | 13,5490 (+0,036) | 1,23e-2 | **1,4** |
| 2048 | 13,5443 | 13,4892 (**−0,055**) | 1,59e-3 | 13,5263 (−0,018) | 8,04e-3 | **5,7** |

**κ varie d'un facteur ~8** (11,7 à T=512, 1,4 à T=1024, 5,7 à T=2048) : Δ = κ·M + c est une **linéarisation locale**, pas une loi à constante universelle. Et **le signe de Δ bascule** : à T=2048 le creux **bat** le dense de 0,055 nat à fraction égale — le bénéfice de retirer le fond diffus (bruit) l'emporte sur le coût de la masse piégée.

Le profil **fin** ρ(d) tranche sur la nature du gradient de récence : ratio ρ(0)/ρ(63) = 0,95 / 1,32 / 1,30, λ ≈ 200–260 clés, R² ≤ 0,32 — une pente **large**, pas une queue de bordure, donc l'approximation « fond plat » tient à l'échelle du trou (1,5 clé), et ρ(0) ≈ 5–6 × 10⁻⁴ est stable sur T = 512/1024/2048.

Réserves : sélection **oracle** (pas l'index PCA déployable), un seul modèle de 135 M, T=2048 sur une seule tranche.
"""
src = src.replace("## La fenêtre est un leurre", add + "\n## La fenêtre est un leurre")
src = src.replace("*Version 22 —", "*Version 23 —")
src = src.replace("`grille_sink.py`", "`grille_sink.py`, `profil_fin.py`, `kappa_T_propre.py`")
out = root / "20_sortie_attention" / "RAPPORT_v23.md"
out.write_text(src, encoding="utf-8")
print("ecrit :", out.name, len(src), "octets")
