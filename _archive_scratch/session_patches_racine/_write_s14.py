# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
add = u"""

### Session 14 — Le sink n'est pas un token special (validation + generalisation)

[QUESTION] Le sink mesure en session 13 etait-il un artefact d'un unique BOS a la position 0 ?

[REPONSE] Non. Trois preuves.
1. SmolLM2-135M n'ajoute AUCUN token special (add_special_tokens=True et False donnent le
   meme token[0]=446 et des sorties identiques). Le sink est donc un token de TEXTE ordinaire.
2. Re-capture independante sur 31 textes locaux du corpus (283 015 caracteres, sans aucun
   token special) : la position 0 porte 30,1 % de la masse, le bloc 0 33,3 %, la fenetre 512
   50,6 %. Sur 4 documents concatenes : 35,6 / 40,3 / 41,5 %.
3. Le correctif (forcer le bloc 0) ramene les tetes catastrophiques a 0,0 % sur les trois
   sequences (7,1 / 13,1 / 26,2 % -> 0,0 %) et l'erreur mediane de 0,072/0,106/0,121 a
   0,048/0,071/0,038 (oracle 0,033/0,049/0,022).

[VALIDATION DU PIPELINE] La replication RoPE a ete verifiee contre les captures existantes
(qk_smol8k.npz) sur les 2048 premieres positions : max|delta K| = 3,906e-03, cosinus =
1,000000 (la RoPE est absolue, donc tronquer la sequence ne change pas ces positions).

[NEW] Le taux d'echec du selecteur par cle moyenne croit avec la longueur du contexte :
7,1 % a 2 048 tokens (wikitext), 26,2 % sur le corpus local a 2 048 tokens, 43-47 % a
8 192 tokens. L'amplitude du sink, elle, est stable (23-36 %).

[FICHIERS] code/capture_local.py ; resultats/capture_local_summary.txt.
"""
with (root / "07_journal_recherche.md").open("a", encoding="utf-8") as f:
    f.write(add)
print("journal session 14 ok")
