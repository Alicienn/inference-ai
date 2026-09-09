# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
add = u"""

### Session 13 — CORRECTION MAJEURE de l'axe 20 : le sink, pas le Jensen gap

[CE QUI A ETE FAIT] Balayage complet des 202 tetes (8 couches x toutes les tetes des deux
modeles), puis comparaison de resumes de passe 1 alternatifs, puis robustesse du selecteur
par cles echantillonnees, puis diagnostic de la masse par position.

[CE QUI ETAIT FAUX] La session 12 concluait a un "Jensen gap catastrophique" du resume par
cle moyenne (43-47 % de tetes avec >0,5 d'erreur de sortie, cas Qwen L11 h7 a 8,57). C'est
un ARTEFACT : ces tetes sont celles dont la masse est au sink d'attention (position 0), que
la cle moyenne dilue et donc ne selectionne pas.

[PREUVE]
1. La masse par position se concentre 31x sur les positions = 0 mod 64 ; la classe est
   dominee a 97 % par la SEULE position 0 (masse 0,296 SmolLM2 / 0,224 Qwen) ; le bloc 0
   porte 0,293 / 0,264 ; la fenetre locale 512 porte 0,429 / 0,526.
2. Forcer le bloc du sink dans le jeu garde : erreur mediane 0,444 -> 0,125 (SmolLM2, 90
   tetes) et 0,346 -> 0,089 (Qwen, 112 tetes) ; tetes catastrophiques 46,7 % -> 2,2 % et
   42,9 % -> 0,0 % ; oracle 0,079 / 0,073. Une ligne de code.
3. Les selecteurs "cle a l'offset 0" (0,172 / 0,108) et coresets r=2/r=4 ne marchaient pas
   pour une autre raison : ils tombaient sur le sink plus souvent. Test d'offset : 0,172
   (offset 0) vs 0,53 (offsets 16/32/48) vs 0,51 (aleatoire).

[CE QUI RESTE VALIDE DE LA SESSION 12]
- La quantification 4 bits est negligeable (0,02 median) et 4 vs 8 bits ne change rien
  (<=12 %) : confirme sur 202 tetes.
- La fenetre locale est necessaire (43-53 % de la masse).
- Aucun resume compact (260-1040 o/bloc) ne bat le fait de garder le sink.

[LECON METHODOLOGIQUE] Un cas pathologique isole (1 tete sur 8) a ete interprete comme un
defaut de mecanisme (Jensen gap sur tete de recuperation) avant que le balayage complet et
le forcage du sink ne montrent qu'il s'agissait d'un token que le banc omettait. Le
balayage complet a coute ~15 minutes et a renverse la conclusion.
"""
with (root / "07_journal_recherche.md").open("a", encoding="utf-8") as f:
    f.write(add)

p = root / "20_sortie_attention" / "RAPPORT_SORTIE.md"
p.write_text(u"""# Axe 20 (v2) — La passe fine n'est pas le goulot, et le vrai goulot s'appelle le sink

*Version 2. La v1 concluait a un "Jensen gap catastrophique" du resume par cle moyenne ; le
balayage complet des tetes et le diagnostic du sink ont montre que c'etait un artefact.*

## Banc

Q/K reels de Qwen3-8B (112 tetes) et SmolLM2-135M (90 tetes), captures 8 192 tokens,
attention causale, fenetre locale exacte 512 + m = 8 blocs de 64 choisis parmi les blocs
anterieurs (~120 candidats), valeurs = cles exactes (proxy V = K). Metrique : erreur L2
relative de la sortie d'attention, moyennee sur 8 requetes de fin.

## 1. La quantification est negligeable, et 4 bits suffisent

quantification 4 bits : mediane 0,020 (max 0,088 / 0,362) ; 8 bits : 0,0005.
Pipeline ASP : 0,444 -> 0,443 (SmolLM2), 0,346 -> 0,352 (Qwen) en passant de 4 a 8 bits,
soit <= 12 % au mieux sur une tete. Le plancher n'est pas la precision de la passe fine.

## 2. Le selecteur de blocs etait le probleme

Avec le selecteur par cle moyenne (coreset r=1, 260 o/bloc) : erreur mediane 0,444 / 0,346,
p90 2,079 / 2,407, tetes catastrophiques (>0,5) 46,7 % / 42,9 %. Oracle (LSE exact) : 0,079
/ 0,073. Aucun resume alternatif ne repare rien : cle de norme maximale 0,61, k-centre r=2
(520 o) 0,51, k-centre r=4 (1040 o) 0,59.

## 3. Tout venait du sink d'attention

Masse par position : concentration 31x sur les positions = 0 mod 64, dominee a 97 % par la
SEULE position 0.

| | SmolLM2 | Qwen |
|---|---|---|
| masse de la position 0 | 0,296 | 0,224 |
| masse du bloc 0 (0-63) | 0,293 | 0,264 |
| masse de la fenetre locale 512 | 0,429 | 0,526 |

La cle moyenne dilue le sink parmi 64 cles, donc ne le selectionne pas : 22-30 % de la masse
perdue d'un coup. Le selecteur par cle a l'offset 0 ne marchait que parce qu'il tombait sur
le sink (test d'offset : 0,172 a l'offset 0 contre 0,53 aux offsets 16/32/48).

## 4. Le correctif tient en une ligne

| methode | err med. SmolLM2 | catastrophes | err med. Qwen | catastrophes |
|---|---|---|---|---|
| cle moyenne | 0,444 | 46,7 % | 0,346 | 42,9 % |
| **+ bloc du sink force** | **0,125** | **2,2 %** | **0,089** | **0,0 %** |
| cle offset 0 | 0,172 | 6,7 % | 0,108 | 1,8 % |
| oracle | 0,079 | 0 % | 0,073 | 0 % |

## Consequences

1. Passe fine a 4 bits (confirme sur 202 tetes).
2. Garder explicitement le bloc du sink : seul correctif qui compte, connu depuis
   StreamingLLM.
3. Fenetre locale exacte obligatoire (43-53 % de la masse).
4. Ne pas conclure d'un cas pathologique a un defaut de mecanisme : le cas Qwen L11 h7
   (erreur 8,57) etait une tete dont la masse est au sink.

## Limites

V = K (quantification de V non modelisee) ; 8 192 tokens, une sequence par modele, 8 requetes
de fin ; oracle = borne superieure ; le sink est la position 0 de la capture ; pas de mesure
de latence (pas de GPU local).

## Reproduction

`code/{err_attention2,qwen_run,sweep_tetes,pass1_compare,pass1_sampled,pass1_offsets,diag_offset,diag_bos,reconciliation_sink}.py` ;
`resultats/` ; verification : cellule 10 du runbook 9944d27c.
""", encoding="utf-8")
print("journal + RAPPORT_SORTIE v2 ok")
