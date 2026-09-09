# Axe 19 — Loi octets-precision des resumes de blocs

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
| quantification par bloc | 3,221 * | 3,290 * | 3,333 * |
| quantification, echelles globales | 2,768 | 2,730 | 2,818 |

A ~2 ko (Qwen/RoPE) : coreset r=16 = 2 080 o, sigma 0,475, rappel 93,7 % ;
COBS r=16 = 2 208 o, sigma 0,925, 84,7 % ; quantification 4 bits = 2 304 o,
sigma 0,099, 99,6 % ; quantification 6 bits = 3 328 o, sigma 0,024, 99,98 %.
Coreset r=64 (exact) = 8 320 o.

\* Ces trois valeurs sont des pentes log-log MOYENNES sur la plage 768-4 352 o, pas des
constantes : ajustees sur des fenetres glissantes de 3 points elles valent 2,49 -> 4,41.
La quantification suit en realite une loi EXPONENTIELLE : sigma = C 2^(-b/(L D/8)), soit
tau = L D/(8 ln 2) = 738,7 o ; mesure 724 / 720 / 728 o (0,975-0,985 du predit) pour k >= 4.
Rapport par bit mesure : 3,51 / 2,35 / 2,16 / 2,08 / 2,04 -> converge vers 2,00.
Consequence : 1 bit de plus par dimension (512 o) divise l'erreur par 2 ; diviser l'erreur
du coreset par 2 coute 12 a 16 fois plus d'octets.

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

## 2bis. Mise a l'echelle en la longueur de bloc L

Prediction de la loi : tau = L·D/(8 ln 2) (lineaire en L) et sigma(4b)/sigma(8b) = 16
pour tout L. Mesure sur Qwen3-8B/RoPE :

| L | tau mesure | tau predit | rapport | sigma(4b)/sigma(8b) |
|---|---|---|---|---|
| 32 | 362,5 o | 369,3 o | 0,981 | 16,86 |
| 64 | 724,3 o | 738,7 o | 0,980 | 16,90 |
| 128 | 1447,2 o | 1477,3 o | 0,980 | 16,95 |

tau est lineaire en L (facteurs 2,00 et 2,00) avec le meme facteur 0,980 sur 4x de plage ;
le rapport 4b/8b est independant de L comme prevu. Cout d'un bit par dimension : L·D/8
octets (512 o a L = 64, 1024 o a L = 128). Fichiers : `code/loi_L.py`,
`resultats/loi_L32_s0.json`, `resultats/loi_L128_s0.json` ; verification : cellule 8 du
runbook 9944d27c.

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
