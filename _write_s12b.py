# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")

add = u"""

### Session 12, axe 20 — la sortie d'attention (fin de l'angle mort)

[OBJECTIF] La loi octets-precision de l'axe 19 mesurait la SELECTION (dispersion du
log-score, rappel de masse). Ce qui compte a l'arrivee est la SORTIE d'attention. Fermer
l'ecart.

[BANC] Q/K reels (Qwen3-8B, SmolLM2-135M), attention CAUSALE, fenetre locale exacte de 512
tokens + m = 8 blocs de 64 tokens choisis parmi tous les blocs entierement anterieurs a la
fenetre (jusqu'a ~120 candidats), valeurs = cles exactes (proxy V = K ; la quantification de
V est un autre sujet). Trois variantes : (a) selection oracle (LSE exact du bloc) ; (b)
quantification seule (attention pleine, cles de score quantifiees) ; (c) pipeline ASP
(passe 1 = cle moyenne, passe 2 quantifiee). Metrique : erreur L2 relative de la sortie.

[RESULTATS]
1. Quantification 4 bits : 0,0004-0,0585 (medianes 0,0029 Qwen / 0,0102 SmolLM2).
   Selection oracle : 0,0014-0,1897 (medianes 0,0076 / 0,0692). La selection coute 2,6 a
   6,8x la quantification au median.
2. Passer de 4 a 8 bits n'ameliore l'erreur du pipeline ASP que de 12 % AU MAXIMUM
   (medianes SmolLM2 0,1256 -> 0,1257). Le plancher d'erreur est la passe grossiere.
3. Sans fenetre locale : erreur de selection 0,012-0,85 (mediane ~0,25) ; la fenetre locale
   porte 0-98 % de la masse (mediane 0,55). Elle n'est pas optionnelle.
4. Cas pathologique Qwen3-8B L11 h7 : masse locale 0,000, oracle 1,000, passe 1 (cle
   moyenne) 0,125, erreur de sortie ASP 8,57 contre 0,0026 pour l'oracle. Mecanisme : la
   cle moyenne est une borne SUPERIEURE de Jensen du LSE du bloc ; sur une tete de
   recuperation (masse concentree sur une cle lointaine) le classement s'inverse. Meme
   famille de defaut que le Jensen gap du compresseur CSA de V4 (-21,8 %).

[CONSEQUENCE] La passe fine doit etre a 4 bits, pas 6-8. L'investissement utile est dans la
passe grossiere (jusqu'a 4,7x sur les tetes ordinaires, 3300x sur la tete pathologique) et
dans le maintien d'une fenetre locale exacte. Ceci corrobore par une metrique independante
(erreur de sortie) la conclusion de la frontiere de Pareto de l'axe 19.

[PIEGE] Une premiere version omettait le masque causal et incluait les blocs recents : la
masse de la passe 1 tombait a 0,001 et le banc semblait condamner ASP. Le resultat etait un
artefact du banc, pas du systeme. Toujours verifier que la masse de la fenetre locale est
coherente avec la litterature (elle l'est : 0,55 en mediane).
"""
with (root / "07_journal_recherche.md").open("a", encoding="utf-8") as f:
    f.write(add)

idx = root / "INDEX.md"
t = idx.read_text(encoding="utf-8")
if "20_sortie_attention" not in t:
    t += u"""

## Axe 20 — Erreur de sortie d'attention (session 12)

`20_sortie_attention/RAPPORT_SORTIE.md` : decomposition de l'erreur de sortie d'attention
sous quantification et sous selection (Qwen3-8B, SmolLM2-135M, masque causal, fenetre
locale exacte). Resultat : la quantification 4 bits coute 2,6-6,8x moins que la selection ;
4 -> 8 bits n'ameliore le pipeline que de <= 12 % ; la cle moyenne echoue sur les tetes de
recuperation (Jensen gap). Code : `20_sortie_attention/code/err_attention2.py`,
`qwen_run.py`. Donnees : `20_sortie_attention/resultats/err_attention_{smol,qwen}_v2.json`.
Verification : cellule 9 du runbook 9944d27c (runbook_run_8ea7e88b).
"""
    idx.write_text(t, encoding="utf-8")
print("journal + INDEX ok")
