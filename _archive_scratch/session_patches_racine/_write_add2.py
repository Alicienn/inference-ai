# -*- coding: utf-8 -*-
import pathlib
root = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
add = u"""

### Addendum 2 Session 12 — la loi tient aussi sur l'axe de la longueur de bloc

[PREDICTION] sigma = C 2^(-b/(L D/8)) implique deux consequences falsifiables :
(i) tau = L D/(8 ln 2) doit etre LINEAIRE en L ; (ii) sigma(4b)/sigma(8b) doit valoir
2^(Delta b / tau) = 2^(L D/2 / (L D/(8 ln2))) = 16 POUR TOUT L (les deux echelles se
compensent). Mesure sur Qwen3-8B/RoPE, L = 32 / 64 / 128, 10 requetes, 4 couches.

[RESULTAT] tau mesure = 362,5 / 724,3 / 1447,2 o ; tau predit = 369,3 / 738,7 / 1477,3 o ;
rapport = 0,981 / 0,980 / 0,980. Le MEME facteur 0,98 sur 4x de plage en L : la forme
fonctionnelle est confirmee, et le facteur 0,98 est une correction constante, pas un effet
d'echelle. sigma(4b)/sigma(8b) = 16,86 / 16,90 / 16,95 (predit 16,00) : independant de L
comme prevu, avec un exces systematique de 5,6 % qui est la contrepartie exacte du deficit
de 2 % sur tau (la decroissance reelle est un peu plus rapide que 2^-k, la composante
log-sum-exp aux bas debits s'ajoutant a l'erreur de quantification pure).

[CONSEQUENCE] Le cout d'un bit par dimension est L D/8 octets : 512 o a L = 64, 1024 o a
L = 128. Des blocs plus longs ne changent donc pas la valeur RELATIVE d'un octet, mais
multiplient proportionnellement le budget necessaire pour atteindre une precision donnee.
Autrement dit, la taille de bloc est un levier neutre sur le rendement et couteux en
absolu : il faut la choisir pour le rappel de selection, pas pour l'economie d'octets.

[VERIFICATION] Cellule 8 du runbook 9944d27c (execution runbook_run_72ae7599) : tau/L
constant a 0,2 % pres, rapport 4b/8b dans (15, 18,5) pour les trois L.
"""
with (root / "07_journal_recherche.md").open("a", encoding="utf-8") as f:
    f.write(add)

rap = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\RAPPORT_OCTETS.md")
t = rap.read_text(encoding="utf-8")
t = t.replace(u"## 3. Limites", u"""## 2bis. Mise a l'echelle en la longueur de bloc L

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

## 3. Limites""")
rap.write_text(t, encoding="utf-8")
print("addendum 2 + rapport axe ok")
