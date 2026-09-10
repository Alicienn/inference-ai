# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\RAPPORT_SORTIE.md")
p.write_text(u"""# Axe 20 — La passe fine n'est pas le goulot

## L'angle mort

L'axe 19 mesurait la SELECTION : dispersion du log-score, rappel de masse @8, frontiere de
Pareto octets/rappel. C'est la bonne metrique pour classer des blocs, pas pour predire la
sortie de l'attention. Ce rapport mesure directement l'erreur relative de la sortie.

## Le banc

Q/K reels de Qwen3-8B et SmolLM2-135M (captures 8 192 tokens), attention CAUSALE, config
realiste d'attention creuse : fenetre locale exacte de 512 tokens + m = 8 blocs de 64 tokens
choisis parmi tous les blocs entierement anterieurs a la fenetre (jusqu'a ~120 candidats).
Valeurs = cles exactes (proxy V = K ; la quantification de V est un sujet distinct, KIVI).

Trois variantes, comparees a l'attention pleine exacte :
1. selection seule : top-8 blocs par LSE exact du bloc (oracle) ;
2. quantification seule : attention pleine, cles de score quantifiees k bits/bloc ;
3. pipeline ASP : passe 1 = cle moyenne du bloc (130 o), passe 2 quantifiee.

Metrique : erreur L2 relative de la sortie, moyennee sur 16 requetes de fin, 8 tetes/modele.

## 1. La quantification est bon marche, la selection est chere

| | SmolLM2-135M | Qwen3-8B |
|---|---|---|
| quantification 4 bits (mediane / max) | 0,0102 / 0,0585 | 0,0029 / 0,0421 |
| selection oracle (mediane / max) | 0,0692 / 0,1137 | 0,0076 / 0,1897 |
| rapport au median | 6,8x | 2,6x |

## 2. Au-dela de 4 bits, il ne se passe rien

Erreur du pipeline ASP complet : SmolLM2 mediane 0,1256 (4 bits) -> 0,1257 (8 bits), soit
0 % ; Qwen3-8B 0,0418 -> 0,0412, soit 1,4 %. Meilleur cas individuel : 12 %. Doubler deux
fois la profondeur de bits ne rachete pas les blocs manquants. Meme conclusion que la
frontiere de Pareto de l'axe 19, par une metrique entierement differente.

## 3. La fenetre locale n'est pas optionnelle

Sans fenetre locale (top-8 sur tous les blocs causals) : erreur de selection 0,012-0,85
(mediane ~0,25). La fenetre porte a elle seule 0-98 % de la masse (mediane 0,55).

## 4. La cle moyenne echoue sur les tetes de recuperation

Qwen3-8B, couche 11, tete 7 :

| | masse captee | erreur de sortie |
|---|---|---|
| fenetre locale | 0,000 | — |
| selection oracle (LSE exact) | 1,000 | 0,0026 |
| passe 1 par cle moyenne | 0,125 | 8,57 |

Mecanisme : la cle moyenne est une borne SUPERIEURE de Jensen du LSE du bloc
(LSE <= log L + q.moyenne(k)) ; l'ecart est maximal quand la masse du bloc se concentre sur
une cle, et le classement s'inverse. Meme famille de defaut que le Jensen gap documente pour
le compresseur CSA de DeepSeek V4 (-21,8 %), mais ici le cout est total sur une tete. Sur les
autres tetes, la passe 1 coute 0 a 4,7x l'erreur de l'oracle.

## Consequences

1. La passe fine doit etre a 4 bits, pas 6-8 : au-dela, les octets ne se convertissent pas en
   qualite.
2. Investir dans la passe grossiere (resume qui borne mieux le LSE, ou score qui n'est pas
   une borne superieure) : jusqu'a 4,7x sur les tetes ordinaires, 3300x sur la tete
   pathologique.
3. Garder une fenetre locale exacte : elle porte la moitie de la masse.

## Limites

- V = K : la quantification de V n'est pas modelisee.
- 8 192 tokens, une sequence par modele, 16 requetes de fin, 8 tetes/modele.
- L'oracle utilise le LSE exact, borne superieure de tout resume compact.
- Aucune mesure de latence (pas de GPU local).

## Reproduction

`code/err_attention2.py` (banc parametrable), `code/qwen_run.py` (replication Qwen) ;
`resultats/err_attention_smol_v2.json`, `resultats/err_attention_qwen_v2.json` ;
verification : cellule 9 du runbook 9944d27c (runbook_run_8ea7e88b).
""", encoding="utf-8")
print("RAPPORT_SORTIE.md ecrit :", p.stat().st_size, "octets")
