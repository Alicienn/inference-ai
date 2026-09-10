# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")

journal = u"""

---

## Session 12 — Axe 19 : la loi octets-precision depend de la FAMILLE de resume

[OBJECTIF] Trancher une question laissee ouverte par le POC : la dispersion du log-score
d'un resume de bloc decroit-elle selon une loi universelle en les octets, ou l'exposant
est-il un choix de conception ?

[FAIT] Sept familles mesurees sur Q/K reels (Qwen3-8B, SmolLM2-135M ; RoPE et NoPE ;
L = 64 ; 8 192 tokens ; 8 couches) : moyenne, COBS rang r, coreset k-centre r, melange
gaussien, sous-echantillonnage, top-norme, quantification scalaire k bits (echelles par
bloc ET globales, avec comptage honnete des octets : `L*D*k/8 + 2*D*2`).

[RESULTAT] L'exposant n'est PAS universel. `sigma_disc ~ b^-alpha` donne alpha = 0,273
(coreset), 0,067 (COBS), 0,544 (sous-echantillonnage), 0,328 (top) pour les familles de
groupement, contre **3,221** (quantification par bloc) et 2,768 (echelles globales).
Robuste sur trois reglages regeneres par le code courant : alpha_quant = 3,221 / 3,290 /
3,333 et alpha_coreset = 0,273 / 0,327 / 0,317. A ~2 ko : coreset r=16 sigma = 0,475 et
93,7 % de rappel ; quant 4 bits sigma = 0,099 et 99,6 %. Le coreset n'atteint
l'exactitude qu'a 8 320 o (bloc brut).

[MECANISME] Les rayons gloutons du k-centre donnent d_r ~ r^-0,21, soit une dimension de
recouvrement d_cov = 4,16 a 5,97 pour une dimension ambiante de 64. La borne du POC
(|l_hat - l| <= s||q||rho) est donc une borne de GROUPEMENT, pas une loi du probleme :
la quantification ne remplace pas les cles par des representants de rayon rho, elle les
approxime a la demi-maille pres, et la maille decroit exponentiellement avec le debit.
Consequence : la quasi-egalite de 0,07 nat EST resolvable (6 bits, 3 328 o, sigma = 0,024).

[RESULTAT] Frontiere de Pareto octets<->rappel de la selection a deux passes
(octets = n*S1 + m1*S2, m = 8, n = 89). Au-dessus de 90 % de rappel, les 14 points de
frontiere utilisent tous une passe 2 QUANTIFIEE. La configuration du papier (passe 2 =
coreset r=16) est dominee : coreset r=1 -> coreset r=16 (m1=4m) = 78 110 o pour 92,77 %,
contre coreset r=2 -> quant 4 bits (m1=2m) = 59 963 o pour 94,34 %. Et la passe 2 exacte
plafonne a 93,30 % pour 142 622 o, la quant 4 bits l'atteint a 0,11 point pres pour
48 414 o (2,95x moins d'octets) ; a m1 = 4m et 8m l'ecart monte a 0,24 et 0,36 point pour
3,21x et 3,37x moins d'octets. Motif reproduit sur Qwen/NoPE et SmolLM2/RoPE.

[LECON] Un test A/B contre le module de reference (`summaries.py`) a revele un bug de
vectorisation : le `lse` supprimait le mauvais axe et ne conservait qu'une seule requete
repete. Le tableau produit etait plausible, monotone, et INVERSAIT l'ordre
coreset/COBS. Deux regles : (1) tout chemin vectorise doit etre compare a
l'implementation de reference sur les memes blocs et les memes requetes ; (2) un resultat
qui contredit le banc precedent est un bug jusqu'a preuve du contraire. Les trois entrees
de `frontiere.json` ecrites par l'ancienne version ont ete regenerees (leur provenance se
detecte a la comptabilite d'octets : `quant 4b = 2 052` ancienne, `2 304` corrigee).

[CE QUE CELA CHANGE POUR LE PAPIER] Le resultat negatif de bout en bout a deux causes
independantes et reparables : (1) le plafond de qualite etait un artefact de la famille
de resume, pas du budget d'octets ; (2) le cout du selecteur etait domine par le nombre
de noyaux (fusion interleaved 1,57x sur H200 a q = 8), pas par les octets. La conception
correcte est donc : selecteur structurel grossier (coreset r=1-2, 130-260 o par bloc)
puis attention sur des cles quantifiees a 4 bits des blocs survivants.

[NON FAIT, honnete] Aucune mesure de latence dans cette session : machine locale sans GPU
et acces distants (vast.ai, Colab) indisponibles. Le rappel de selection n'est pas
l'erreur de sortie d'attention. Captures a 8 192 tokens, une graine par reglage, aucun
modele entraine nativement avec attention creuse.

[PROCHAIN TEST] Integrer la passe 2 quantifiee dans `16_gpu/e2e_qwen.py` et refaire la
mesure de bout en bout avec le noyau fusionne, pour separer les deux causes (qualite vs
noyaux). C'est le seul test qui manque pour savoir si ASP est viable.
"""
with (root / "07_journal_recherche.md").open("a", encoding="utf-8") as f:
    f.write(journal)
print("journal ok, taille:", (root / "07_journal_recherche.md").stat().st_size)

rap = u"""# Axe 19 — Loi octets-precision des resumes de blocs

Resume des mesures. Details complets et figures : rapport de workspace
`report_56cfaa71-7eb8-4883-b2d0-d972ed4f97ad` ; runbook
`runbook_9944d27c-84ea-4878-bab6-c95ff9b00607`.

## 1. L'exposant depend de la famille

`sigma_disc ~ b^-alpha`, Q/K reels, L = 64, 8 192 tokens.

| famille | alpha (Qwen/RoPE) | alpha (Qwen/NoPE) | alpha (SmolLM2/RoPE) |
|---|---|---|---|
| coreset (k-centre) | 0,273 | 0,327 | 0,317 |
| COBS | 0,067 | 0,087 | 0,118 |
| sous-echantillonnage | 0,544 | 0,624 | 0,354 |
| top-norme | 0,328 | 0,421 | 0,152 |
| quantification par bloc | 3,221 | 3,290 | 3,333 |
| quantification, echelles globales | 2,768 | 2,730 | 2,818 |

A ~2 ko (Qwen/RoPE) : coreset r=16 = 2 080 o, sigma 0,475, rappel 93,7 % ;
COBS r=16 = 2 208 o, sigma 0,925, 84,7 % ; quantification 4 bits = 2 304 o,
sigma 0,099, 99,6 % ; quantification 6 bits = 3 328 o, sigma 0,024, 99,98 %.
Coreset r=64 (exact) = 8 320 o.

Dimension de recouvrement mesuree par les rayons gloutons du k-centre :
d_cov = 4,75 / 4,16 / 5,97 contre D = 64. C'est elle qui fixe la loi polynomiale des
familles de groupement ; la quantification y echappe (maille decroissante en 2^-k).

## 2. Frontiere de Pareto de la selection a deux passes

octets = n*S1 + m1*S2 ; m = 8 ; n = 89 (Qwen/RoPE).

| octets | rappel @8 | configuration |
|---|---|---|
| 11 550 | 85,26 % | plat coreset r=1 |
| 19 870 | 87,68 % | coreset r=1 -> coreset r=4 (m1=2m) |
| 28 190 | 89,18 % | coreset r=1 -> coreset r=8 (m1=2m) |
| 32 030 | 90,47 % | coreset r=1 -> quant 2 bits (m1=2m) |
| 43 579 | 91,53 % | coreset r=2 -> quant 2 bits (m1=2m) |
| 48 414 | 93,19 % | coreset r=1 -> quant 4 bits (m1=2m) |
| 59 963 | 94,34 % | coreset r=2 -> quant 4 bits (m1=2m) |
| 85 278 | 97,35 % | coreset r=1 -> quant 4 bits (m1=4m) |
| 159 006 | 99,29 % | coreset r=1 -> quant 4 bits (m1=8m) |
| 204 696 | 99,60 % | plat quant 4 bits |
| 295 672 | 99,98 % | plat quant 6 bits |

Au-dessus de 90 % de rappel, 14/14 points de frontiere ont une passe 2 quantifiee.
Passe 2 exacte : 93,30 % pour 142 622 o ; quant 4 bits : 93,19 % pour 48 414 o.

## 3. Limites

Aucune mesure de latence (pas de GPU local, acces distants indisponibles). Le rappel
mesure la selection, pas l'erreur de sortie d'attention. Captures 8 192 tokens, une
graine par reglage, aucun modele nativement creux.

## 4. Fichiers

- `code/frontiere2.py` : frontiere octets-precision (7 familles), rayons gloutons.
- `code/asp_pareto3.py` : frontiere de Pareto a deux passes.
- `code/ab_test.py` : test A/B contre `12_poc/code/summaries.py` (a refaire a chaque
  modification du chemin vectorise).
- `resultats/frontiere.json`, `resultats/asp_pareto.json`.
"""
(root / "19_loi_octets" / "RAPPORT_OCTETS.md").write_text(rap, encoding="utf-8")
print("rapport axe ok:", (root / "19_loi_octets" / "RAPPORT_OCTETS.md").stat().st_size)
