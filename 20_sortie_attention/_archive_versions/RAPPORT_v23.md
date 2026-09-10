# Le sélecteur d'ASP était le problème, pas son idée

*Version 23 — trois défauts corrigés, trois architectures, une loi quantitative, et une structure d'attention vérifiée*

## En une phrase

L'attention creuse par sélection de blocs n'échoue pas parce que l'idée est mauvaise, mais parce que le sélecteur utilisé (un résumé par clé moyenne) détruit le signal. Avec le bon estimateur — un **maximum** sur des projections PCA à 8 dimensions, base estimée sur les 256 premiers tokens — et la bonne granularité de blocs, un quart du contexte suffit pour rester à quelques centièmes du dense. La perte de granularité suit **Δ ≈ κ·ρ·(Lb−1)/2**, et tout s'éclaire une fois l'attention hors fenêtre décomposée : **un sink de début de séquence (≈70 %) + un fond plat (≈30 %)**, structure vérifiée sur trois architectures et sur la grille (T,W).

## Trois défauts, et pourquoi ils comptent

**1 — Fuite du futur dans le masque.** Masques creux omettant la borne causale `j ≤ p` : le modèle lisait les tokens suivants en s'échappant par la fenêtre locale, ce qui inversait le classement des sélecteurs. Corrigé ; invariance par préfixe à 1,6–4,1 × 10⁻⁵.

**2 — Le sentinelle qui offrait le *sink*.** Trouvé par **vérification indépendante** : la valeur `-1` (« aucun bloc éligible ») était ramenée à l'indice 0, marquant le bloc 0 — qui contient le *sink* — comme sélectionné dès qu'il y avait moins de *m* blocs éligibles. Le bug **flattait tous les sélecteurs creux**. 23 scripts corrigés ; contrôle structurel 101/192.

**3 — La base voyait le futur.** Base PCA estimée sur **tout** le segment : index transductif, non déployable. Audit apparié :

| | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| PCA d′=8, transductive | +0,0323 | +0,0447 |
| PCA d′=8, **causale** | **+0,0107** | **+0,0222** |
| PCA d′=16, causale | **+0,0049** | **−0,0002** |

Voir le futur **nuit** de ~0,022 nat. Le chiffre publié était pessimiste *et* non déployable.

## Et une quatrième passe qui, elle, confirme

Réimplémentation indépendante étendue à Qwen2.5-0.5B (GQA 14/2), au sélecteur PCA causal et au RoPE : base par **SVD**, tri explicite au lieu de `topk`, tête KV par `h // 7`. Sur **84 comparaisons** : blocs sélectionnés identiques **84/84**, masques identiques **84/84**, sorties à **1,4 × 10⁻⁵**.

## La granularité, levier dominant

**Mécanisme** : avec une fenêtre W, une grille de taille Lb laisse un trou de (p−W+1) mod Lb clés — en moyenne **(Lb−1)/2** — entre la fenêtre et le bloc éligible le plus proche.

Test à budget constant (m·Lb = 64, W = 128 → 37,5 % des clés) :

| Lb | m | SmolLM2 | Qwen2.5-0.5B | GPT-2 |
|---|---|---|---|---|
| 64 | 1 | +0,385 | +0,802 | +0,400 |
| 32 | 2 | +0,215 | +0,346 | +0,216 |
| 4 | 16 | **+0,021** | **−0,003** | **−0,004** |
| 1 | 64 | — | **−0,004** | — |

**La courbe sature à Lb=4**, où le sélecteur égale l'oracle par clé.

## La loi quantitative

Mesure directe sur les distributions d'attention exactes (p ≥ 192 pour que le trou ne puisse pas atteindre la position 0) :

| Lb | trou | masse piégée | perte (SmolLM2) | perte/masse |
|---|---|---|---|---|
| 4 | 1,5 | 0,00111 | 0,0208 | 18,7 |
| 8 | 3,5 | 0,00234 | 0,0594 | 25,4 |
| 16 | 7,5 | 0,00651 | 0,1145 | 17,6 |
| 32 | 15,5 | 0,01391 | 0,2145 | 15,4 |
| 64 | 31,5 | 0,03133 | 0,3848 | 12,3 |

**Δloss = 11,68 × (masse piégée) + 0,030, R² = 0,986.**

**Le test qui tranche** — mêmes mesures sur Qwen2.5-0.5B et GPT-2, constantes de SmolLM2 appliquées telles quelles :

| modèle | κ (nat/masse) | ρ (par clé) | κρ | Lb_max prédit (ε = 0,02) |
|---|---|---|---|---|
| SmolLM2-135M | 11,7 | 7,80 × 10⁻⁴ | 0,0091 | 5,4 |
| GPT-2 | 17,9 | 7,52 × 10⁻⁴ | 0,0135 | 4,0 |
| Qwen2.5-0.5B | 33,4 | 8,27 × 10⁻⁴ | 0,0276 | **2,5** |

**La forme se généralise, l'échelle non** : R² = 0,999 (Qwen) et 0,992 (GPT-2) ; ρ quasi invariante (< 10 % d'écart) ; κ varie d'un facteur 2,9. L'optimum prédit par modèle (5,4 / 4,0 / 2,5) encadre le Lb=4 observé et ordonne correctement les modèles.

## La structure qui explique tout — et sa vérification

Première tentative : un plancher uniforme, ρ = (1−μ_fenêtre)/(T−W). La densité juste derrière la fenêtre vaut 0,18 à 0,49 fois la moyenne hors fenêtre, ce qui ressemble à un creux. **La décomposition le résout.** Distribution conditionnelle hors fenêtre (T=512, W=128) :

| modèle | masse hors fenêtre | support effectif | masse top-1 | argmax dans les 10 1res | 1er décile | déciles 2–10 |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 0,582 | 15,3 | 0,668 | **0,965** | 0,705 | plats (~0,03) |
| GPT-2 | 0,490 | 35,4 | 0,634 | **0,865** | 0,659 | plats (~0,04) |
| Qwen2.5-0.5B | 0,460 | 20,6 | 0,667 | **0,806** | 0,688 | plats (~0,03) |

**L'argmax de l'attention hors fenêtre tombe dans les dix premières positions de la séquence** pour 81 à 97 % des requêtes, sur trois architectures dont une sans RoPE : c'est bien le *sink*, et non « quelque part dans le premier décile ». La clé dominante porte 63 à 67 % de la masse hors fenêtre ; le premier décile en porte 66 à 71 % ; les neuf déciles suivants sont **plats**. La masse hors fenêtre elle-même varie (0,46 à 0,58), le support effectif aussi (15 à 35 clés), mais **la forme sink + fond ne varie pas**.

Le « creux » annoncé était donc le fond plat **comparé à une moyenne gonflée par le sink** : l'hypothèse du plancher uniforme était juste pour la composante de fond, et ρ mesure celle-ci.

**Pourquoi la sélection marche.** Le sink vit dans le **bloc le plus ancien**, toujours éligible, donc **jamais piégé dans le trou de bordure** : le trou ne prend jamais que de la masse de fond. Et le fond étant plat, classer les blocs de fond est un problème de **quasi-égalité** — exactement les écarts de 0,07 nat mesurés dans le POC initial. La loi de granularité et le résultat négatif du POC sont deux vues de la même structure.


## Ce que le contexte long change

La décomposition répétée sur la grille (T, W) révèle une troisième composante :

| T | W | masse hors fenêtre | support effectif | top-1 | argmax dans les 10 1res | profil des 10 déciles |
|---|---|---|---|---|---|---|
| 512 | 128 | 0,552 | 15,4 | 0,675 | **0,966** | 0,714 / 0,034 / 0,044 / … / 0,035 |
| 1024 | 128 | 0,563 | 30,1 | 0,655 | **0,965** | 0,716 / 0,056 / 0,031 / … / 0,041 |
| 2048 | 128 | 0,588 | 59,4 | 0,568 | **0,917** | 0,633 / 0,041 / 0,026 / 0,018 / 0,019 / 0,034 / 0,040 / 0,047 / 0,060 / **0,082** |
| 2048 | 256 | 0,533 | 52,9 | 0,608 | **0,917** | 0,676 / 0,044 / 0,028 / … / **0,060** |

Le sink tient à toutes les longueurs — 92 à 97 % des argmax restent dans les dix premières positions — mais **sa part baisse** (0,71 → 0,63) et, à T=2048, **le fond n'est plus plat** : minimum aux déciles 3–4 (0,018) puis **remontée monotone jusqu'à 0,082 au dernier décile**, c'est-à-dire juste derrière la fenêtre. C'est un **gradient de récence**, et il place le trou de bordure dans la zone **la plus dense** du fond — le pire endroit pour perdre des clés.

La pénalité mesurée reste néanmoins faible (0,004 nat à T=2048 en configuration mise à l'échelle), donc le gradient **déplace le mécanisme sans l'inverser**. Le support effectif double quand T double (15 → 30 → 59 clés).


## La constante n'est pas constante

À fraction lue constante (37,5 %, W=T/8, m·Lb=T/4), avec la masse piégée **mesurée** dans le même passage :

| T | dense | Lb=4 (Δ) | M₄ | Lb=16 (Δ) | M₁₆ | κ |
|---|---|---|---|---|---|---|
| 1024 | 13,5125 | 13,5350 (**+0,022**) | 2,52e-3 | 13,5490 (+0,036) | 1,23e-2 | **1,4** |
| 2048 | 13,5443 | 13,4892 (**−0,055**) | 1,59e-3 | 13,5263 (−0,018) | 8,04e-3 | **5,7** |

**κ varie d'un facteur ~8** (11,7 à T=512, 1,4 à T=1024, 5,7 à T=2048) : Δ = κ·M + c est une **linéarisation locale**, pas une loi à constante universelle. Et **le signe de Δ bascule** : à T=2048 le creux **bat** le dense de 0,055 nat à fraction égale — le bénéfice de retirer le fond diffus (bruit) l'emporte sur le coût de la masse piégée.

Le profil **fin** ρ(d) tranche sur la nature du gradient de récence : ratio ρ(0)/ρ(63) = 0,95 / 1,32 / 1,30, λ ≈ 200–260 clés, R² ≤ 0,32 — une pente **large**, pas une queue de bordure, donc l'approximation « fond plat » tient à l'échelle du trou (1,5 clé), et ρ(0) ≈ 5–6 × 10⁻⁴ est stable sur T = 512/1024/2048.

Réserves : sélection **oracle** (pas l'index PCA déployable), un seul modèle de 135 M, T=2048 sur une seule tranche.

## La fenêtre est un leurre

| W | m | clés | frac | SmolLM2 | GPT-2 |
|---|---|---|---|---|---|
| 32 | 40 | 192 | 37,5 % | +0,027 | — |
| 128 | 16 | 192 | 37,5 % | +0,021 | — |
| 64 | 16 | 128 | 25 % | +0,037 | +0,020 |
| **256** | **0** | 256 | 50 % | **+1,234** | — |
| **128** | **0** | 128 | 25 % | **+1,677** | **+1,423** |

**La répartition fenêtre/quota ne compte presque pas** (0,008 nat) ; **la localité seule ne vaut presque rien** (facteur 50 à 70 à octets identiques).

## L'avantage grandit avec le contexte

Configuration mise à l'échelle avec T (fraction lue constante à 25 % : W=T/8, m=T/32 blocs de Lb=4) :

| T | clés | vs dense | fenêtre pure | ratio |
|---|---|---|---|---|
| 512 | 128 | +0,041 | +1,763 | 43× |
| 1024 | 256 | +0,012 | +1,241 | 108× |
| 2048 | 512 | **+0,004** | +1,076 | **299×** |

## Généralité : trois architectures

| | SmolLM2-135M | Qwen2.5-0.5B | GPT-2 |
|---|---|---|---|
| RoPE | oui | oui | **non** |
| GQA | 9/3 | 14/2 | 12/12 |
| granularité (facteur) | 18 | 240 | ~100 |
| fenêtre pure (facteur) | 50 | — | 70 |
| κ | 11,7 | 33,4 | 17,9 |

## Déployabilité

- **Base gelée** sur les 256 premiers tokens : +0,0032 nat, *mieux* que la recalculer (+0,0551).
- **Transfert** inter-domaines : 0,007 à 0,075 nat contre 1,52 à 2,05 pour le hasard (20–200 ×).
- **Projection aléatoire** : +0,55 nat. La base doit être adaptée.
- **Index appris** : +0,0964 contre +0,0770 pour la PCA hors échantillon — mais il bat la PCA **en** échantillon. Sur-apprentissage, pas manque de capacité.

## État

Restent, tous hors du local : **la conversion en latence** (RTX 5090 + modèle 7–8B), **les valeurs V réelles** (le banc prend V = K), et **un modèle nativement sparse long-contexte**. Réglage candidat : **Lb=4, W=T/8, m=T/32** (25 % des clés), index PCA causal d′=8, base gelée sur 256 tokens.

## Reproductibilité

`20_sortie_attention/code/` : `courbe_corrigee.py`, `remesure_qwen.py`, `echelle_corrigee.py`, `transfert_domaines.py`, `base_figee.py`, `index_appris_v2.py`, `audit_causal_base.py`, `balayage_Lb_m.py`, `balayage_budget_constant.py`, `balayage_fenetre.py`, `frontiere_budget.py`, `gpt2_generalite.py`, `echelle_saturee.py`, `mecanisme_trou.py`, `mecanisme_corrige.py`, `invariance_kappa.py`, `plancher_uniforme.py`, `decomposition_distante.py`, `sink_verif.py`, `grille_sink.py`, `profil_fin.py`, `kappa_T_propre.py`, `verification_v2.py`, `verification_qwen.py`, `diag_verif.py`. Sorties dans `resultats/`.
