# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")

add = u"""

### Addendum Session 12 — la loi n'est pas une loi de puissance

[OBJECTIF] Tester une prediction qui departage deux hypotheses concurrentes sur la
famille quantification : vraie loi de puissance (pente log-log constante) ou loi
exponentielle locale (pente log-log ~ b/tau).

[DERIVATION] Quantifieur uniforme de pas Delta par bloc : ||delta|| <= sqrt(D) Delta/2,
Delta = (max-min)/2^k, donc sigma ~ 2^-k. Or k = 8(b - 4D)/(L D), lineaire en les octets.
D'ou sigma = C 2^(-b/(L D/8)), de constante de decroissance en logarithme naturel
tau = L D/(8 ln 2) = 738,7 o pour L = D = 64.

[TESTS, tous concluants] (a) rapport par bit -> 2,00 : mesure 3,51 / 2,35 / 2,16 / 2,08 /
2,04 (de 1->2 bits a 6->8 bits). (b) pente log-log locale mesuree -2,46 / -2,54 / -3,06 /
-3,87 / -5,24 contre b/tau predit -1,39 / -2,08 / -2,77 / -3,81 / -5,20 : accord a 1,5 %
sur les deux derniers points, ecart attendu sur les 1-3 bits. (c) alpha ajuste sur fenetres
glissantes de 3 points : 2,49 -> 2,75 -> 3,57 -> 4,41 (Qwen/RoPE), 2,64 -> 4,44 (NoPE),
2,70 -> 4,40 (SmolLM2) : une vraie loi de puissance exigerait un alpha constant.
(d) constante mesuree : 724 / 720 / 728 o contre 738,7 predits (0,975-0,985) pour k >= 4.

[CORRECTION] L'enonce « alpha_quant = 3,2, stable » de la premiere passe est remplace par :
la quantification suit une loi exponentielle en le debit, et 3,22 est la pente locale
moyenne sur la plage 768-4 352 o. Les familles de groupement, elles, suivent bien une loi
de puissance (alpha 0,155 a 0,49 selon la plage, borne par d_cov = 4,2-6,0 contre D = 64).

[CONSEQUENCE PRATIQUE] Un octet par dimension en plus (512 o par bloc de 64 tokens) divise
l'erreur par 2, sans rendement decroissant ; diviser l'erreur du coreset par 2 coute 12 a
16 fois plus d'octets. Le rendement marginal d'un octet est 13 a 18 fois plus eleve pour la
quantification, et l'ecart se creuse avec le budget.

[LECON] Ajuster une loi de puissance sur 6 points et 2 decades est une operation sans
garantie : il faut tester la constance de l'exposant sur des sous-fenetres avant de le
presenter comme une constante du probleme. Le test coute 10 lignes.
"""
with (root / "07_journal_recherche.md").open("a", encoding="utf-8") as f:
    f.write(add)

rap = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\RAPPORT_OCTETS.md")
t = rap.read_text(encoding="utf-8")
t = t.replace(u"| quantification par bloc | 3,221 | 3,290 | 3,333 |",
              u"| quantification par bloc | 3,221 * | 3,290 * | 3,333 * |")
t = t.replace(u"Dimension de recouvrement mesuree par les rayons gloutons du k-centre :",
              u"""\\* Ces trois valeurs sont des pentes log-log MOYENNES sur la plage 768-4 352 o, pas des
constantes : ajustees sur des fenetres glissantes de 3 points elles valent 2,49 -> 4,41.
La quantification suit en realite une loi EXPONENTIELLE : sigma = C 2^(-b/(L D/8)), soit
tau = L D/(8 ln 2) = 738,7 o ; mesure 724 / 720 / 728 o (0,975-0,985 du predit) pour k >= 4.
Rapport par bit mesure : 3,51 / 2,35 / 2,16 / 2,08 / 2,04 -> converge vers 2,00.
Consequence : 1 bit de plus par dimension (512 o) divise l'erreur par 2 ; diviser l'erreur
du coreset par 2 coute 12 a 16 fois plus d'octets.

Dimension de recouvrement mesuree par les rayons gloutons du k-centre :""")
rap.write_text(t, encoding="utf-8")
print("addendum journal + correction rapport axe ok")
