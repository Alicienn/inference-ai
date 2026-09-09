# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
add = u"""

### Session 15 — Le selecteur de blocs est pire que le hasard (perplexite)

[HARNAIS] L'attention de SmolLM2-135M est remplacee par un forward propre (fenetre locale W
+ m blocs de Lb choisis, o_proj conserve). Validation : le mode dense patche reproduit la
perte native a l'identique (2,7564 / 2,6260). T=1024, Lb=64, corpus local.

[RESULTATS] dense 2,7564 (ppl 15,74) ; W256 m0 (fenetre seule) 3,7909 (44,30) ;
W256 m4 3,9240 (50,60) ; W256 m8 4,0116 (55,24) ; W256 m4 ALEATOIRE 3,8634 (47,63) ;
W512 m4 4,0235 (55,89).

[LECTURE]
1. La fenetre seule coute +1,0345 nat, soit 67 % de la penalite totale (+1,2558).
2. La degradation est monotone en m : chaque bloc ajoute par le selecteur empire.
3. LE SELECTEUR PAR CLE MOYENNE EST PIRE QUE LE HASARD : 3,9240 contre 3,8634 (graine 1234).
   Il choisit activement les mauvais blocs.
4. Anomalie a verifier : W512_m4 (4,0235) est pire que W256_m4 (3,9240) malgre 768 tokens
   visibles contre 512.

[CONSEQUENCE] La passe 1 par resume compact n'apporte aucune valeur au niveau modele dans
cette configuration ; le forcage du sink ne recuperait que 4 % de la penalite (session 14) ;
et la metrique par tete surestimait l'importance du sink. Le goulot est la SELECTION.

[LIMITES] une tranche de 1024 tokens, un seul modele, un seul tirage aleatoire (la
difference de 0,06 nat n'est pas encore separee du bruit de tirage), V = K.

[FICHIERS] code/perplexite_configs.py ; resultats/perplexite_configs.txt.
"""
with (root / "07_journal_recherche.md").open("a", encoding="utf-8") as f:
    f.write(add)
print("journal session 15 ok")
