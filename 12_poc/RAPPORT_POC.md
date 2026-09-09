# POC — Résumés de blocs pour l'attention creuse : de l'espace des moments à l'espace du support

**Date :** 30 août 2026 · **Code :** [`code/`](code/) · **Sorties brutes :** [`resultats/_brut/`](resultats/_brut/)
**Bancs d'essai :** Qwen2.5-0.5B et SmolLM2-135M, Q/K réels capturés sur 8192 tokens (wikitext), CPU seul.

---

## 0. Résumé

Ce POC part d'une question précise : **quelle information faut-il stocker par bloc de KV cache pour
décider quels blocs lire ?** C'est le maillon critique de toute attention creuse par blocs, et
l'état de l'art le plus récent (COBS, [arXiv 2607.09052](https://arxiv.org/abs/2607.09052), juillet
2026) y répond par un développement en cumulants tronqué à l'ordre 2 : moyenne + covariance.

Je propose un cadre différent, je le teste, et j'obtiens **un résultat positif robuste et quatre
résultats négatifs** — ces derniers convergeant tous vers une même cause structurelle qui, à mon
sens, est la découverte la plus importante de ce travail.

| | Résultat |
|---|---|
| ✅ **Positif** | Un résumé par **coreset** (r centroïdes + effectifs, scoring log-sum-exp) domine le résumé par covariance à budget d'octets égal, sur 2 modèles × 3 tailles de bloc × 2 régimes RoPE. À L=64 avec RoPE : coreset r=2 (coût 2,03) fait **87,2 %** de rappel de masse contre **82,8 %** pour COBS r=8 (coût 9,12) — mieux pour **4,5× moins cher**. |
| 🔬 **Mécanisme** | La covariance intra-bloc a un **rang effectif de 17,8 sur 64** : elle n'est pas de rang faible. Un résumé de rang 4 n'en capte que 53,5 % ; la fraction captée varie d'un bloc à l'autre, ce qui injecte du bruit dans le classement. Le bloc est **groupé**, pas de **rang faible**. |
| ❌ **Négatif ×4** | Quadrature à poids ajustés, certificats de correction, calibration apprise, budget adaptatif : les quatre échouent à améliorer le classement. |
| 💡 **Cause commune** | **Les blocs sont quasi ex æquo.** L'écart de log-masse entre le 8ᵉ et le 9ᵉ bloc vaut **0,070 nat**, contre une incertitude résiduelle de 1,4 à 2,6 nats — 20 à 37× plus. Aucun estimateur compact ne peut résoudre l'ordre près de la coupure. |

---

## 1. Le cadre : résumer un bloc, c'est quantifier une mesure

Un bloc de KV cache n'est rien d'autre qu'une **mesure empirique** sur l'espace des clés :

```
mu_b = somme_{j dans b} delta_{k_j}
```

et sa masse d'attention est exactement la **transformée de Laplace** de cette mesure :

```
m_b(q) = integrale e^{s q.k} d mu_b(k)          avec s = 1/sqrt(D)
```

Résumer un bloc, c'est donc remplacer `mu_b` par une mesure à peu d'atomes dont la transformée de
Laplace coïncide sur la loi des requêtes. **C'est un problème de quadrature à noyau exponentiel**,
où les requêtes jouent le rôle des fonctions test.

Ce point de vue unifie tout l'état de l'art et fait apparaître **deux familles disjointes** :

| Famille | Ce qu'on approxime | Méthodes |
|---|---|---|
| **Espace des moments** | le comportement *local* de la transformée en q=0 (ses cumulants) | mean-pool (ordre 1), **COBS** (ordre 2) |
| **Espace du support** | la *géométrie* du nuage de clés | Quest (boîte englobante), **coreset** (proposé ici) |

En posant `l_b(q) = log m_b(q)`, l'identité variationnelle de Gibbs donne

```
l_b(q) = max_{p dans le simplexe} [ somme_r p_r (s q.k_r) + H(p) ]
```

`l_b` est donc un **max adouci** : il interpole entre `log L + moyenne` quand `q -> 0` et
`max_r s q.k_r` quand `||q|| -> infini`. Les deux familles approximent ce même objet aux deux
extrémités opposées de son domaine.

### Un théorème pour la famille du support

> **Proposition.** Soit un bloc dont les clés sont partitionnées en `C_1..C_r`, de centroïdes
> `mu_c`, d'effectifs `n_c` et de rayons `rho_c = max_{k dans C_c} ||k - mu_c||`. Alors, **pour
> toute requête q** :
>
> ```
> | l_b(q) - log somme_c n_c e^{s q.mu_c} |  <=  s ||q|| max_c rho_c        (*)
> ```
>
> **Preuve.** Pour chaque `c` et chaque `k` de `C_c`, Cauchy-Schwarz donne
> `|q.(k - mu_c)| <= ||q|| rho_c`, d'où l'encadrement
> `n_c e^{s q.mu_c - s||q||rho_c} <= somme_{k dans C_c} e^{s q.k} <= n_c e^{s q.mu_c + s||q||rho_c}`.
> En sommant sur `c` puis en prenant le logarithme, les facteurs `e^{±s||q||rho}` se factorisent. ∎

Trois propriétés que le développement en cumulants n'a pas :
1. **Uniforme en direction de q** — pas un contrôle local autour de 0 ;
2. **Signée** — `e^{q.mu_c}` distingue `+mu` de `-mu`, là où `q^T Sigma q` est aveugle au signe
   (faiblesse que COBS constate elle-même, §7.7) ;
3. **Exacte sur une aiguille isolée**, qui forme un singleton de rayon nul.

**Vérification numérique :** la borne (*) est respectée dans **100,00 %** des cas mesurés.
Elle est en revanche très lâche (l'erreur réelle ne vaut que 7,2 % de la borne) — j'y reviens en §5.

Par ailleurs, **l'inégalité de Jensen rend la borne inférieure gratuite et exacte** :
`(1/n_c) somme_j e^{q.delta_j} >= 1`, donc le coreset **sous-estime toujours** : `l_b >= lhat_b`.

---

## 2. Le résultat positif : le support bat les moments

**Protocole.** Q/K réels post-RoPE. Comme dans tout système bloc-creux réel (NSA, CSA, COBS), la
fenêtre locale et le puits d'attention sont servis par des branches séparées : je les **exclus donc
des candidats** pour ne mesurer que la récupération *distante*. Sans cette exclusion la métrique
est dominée par des blocs triviaux — et c'est ce qui faisait paraître Quest excellent (89 %) alors
qu'il s'effondre à 69 % une fois le protocole corrigé.

**Coût** exprimé en nombre de vecteurs de dimension D stockés par bloc (un scalaire = 1/D).

### Qwen2.5-0.5B, L=64, avec RoPE — rappel de masse

| méthode | coût | RMSE | @1 | @2 | @4 | **@8** |
|---|---:|---:|---:|---:|---:|---:|
| mean-pool | 1,00 | 7,43 | 76,93 % | 78,84 % | 81,60 % | 85,17 % |
| quest | 2,00 | 20,97 | 52,29 % | 56,56 % | 62,28 % | 69,47 % |
| cobs r=1 | 2,02 | 6,53 | 65,05 % | 69,04 % | 74,39 % | 80,71 % |
| **coreset r=2** | **2,03** | **2,97** | **78,88 %** | **80,50 %** | **83,74 %** | **87,20 %** |
| cobs r=4 | 5,06 | 5,31 | 66,49 % | 70,81 % | 75,94 % | 81,37 % |
| **coreset r=4** | **4,06** | **2,49** | **80,96 %** | **83,02 %** | **85,92 %** | **88,87 %** |
| cobs r=8 | 9,12 | 4,90 | 68,13 % | 72,08 % | 77,28 % | 82,80 % |
| **coreset r=8** | **8,12** | **1,94** | **83,51 %** | **85,54 %** | **88,08 %** | **90,80 %** |

**Le coreset r=2 (coût 2,03) bat COBS r=8 (coût 9,12) sur toutes les métriques.**

### Robustesse

| Configuration | coreset r=2 @8 | cobs r=8 @8 | mean @8 |
|---|---:|---:|---:|
| Qwen, L=32, RoPE | 84,44 % | 78,49 % | 81,96 % |
| Qwen, L=64, RoPE | 87,20 % | 82,80 % | 85,17 % |
| Qwen, L=128, RoPE | 90,24 % | 88,17 % | 88,83 % |
| Qwen, L=64, NoPE | 87,91 % | 85,79 % | 86,52 % |
| **SmolLM2, L=64, RoPE** | **82,56 %** | 78,05 % | 78,65 % |
| **SmolLM2, L=64, NoPE** | **87,72 %** | 85,52 % | 84,58 % |

Le classement est **invariant** sur 2 modèles × 3 tailles de bloc × 2 régimes.

**Observation gênante pour COBS, à signaler honnêtement :** dans mon protocole, COBS fait *moins
bien que le simple mean-pool* dans presque toutes les configurations. La seule exception est
L=128 sans RoPE (90,68 % contre 89,62 %) — c'est-à-dire **exactement le régime pour lequel COBS a
été conçue** (grands blocs, branche de sélection en NoPE). Cela suggère que le gain de COBS est
réel mais étroitement conditionné à son cadre, ce que ma §7 discute.

### Le mécanisme : rang effectif, pas ordre du développement

Ma première hypothèse était que le développement en cumulants échoue parce qu'on serait en régime
de **grandes déviations** (série asymptotique divergente). **Les données l'ont réfutée** : le reste
après l'ordre 2 ne vaut que 0,21 fois le terme d'ordre 2. L'ordre 2 est, en soi, une bonne
approximation.

Le vrai coupable est le **rang** :

| | rang 1 | rang 2 | rang 4 | rang 8 | rang 16 | rang 32 |
|---|---:|---:|---:|---:|---:|---:|
| variance cumulée captée | 21,4 % | 34,9 % | 53,5 % | 74,4 % | 90,3 % | 98,2 % |

**Rang effectif de la covariance intra-bloc : 17,8 sur 64** (entropie de participation). Il faut
17 directions pour capter 90 % de la variance. Un résumé de rang 4 n'attrape que la moitié du terme
quadratique — et **cette fraction varie d'un bloc à l'autre**, ce qui est exactement du bruit de
classement.

> **La structure d'un bloc de clés est *groupée*, pas de *rang faible*.** C'est pourquoi r
> centroïdes valent bien mieux que r directions propres : ils épousent la bonne structure.

---

## 3. Quatre pistes explorées, quatre échecs instructifs

J'ai essayé quatre idées. Toutes améliorent quelque chose, aucune n'améliore le classement.

### Concept 1 — EQS : quadrature à poids ajustés
La théorie de la quadrature dit que r nœuds à **poids libres** matchent bien plus de moments que r
nœuds à poids imposés (idée de Gauss : exactitude à l'ordre 2r−1). J'ajuste donc les poids `w_c` par
moindres carrés sur la loi des requêtes (calibration sur la **loi**, pas sur un point q₀ — ce qui
évite l'écueil de l'expansion query-centered que COBS a testée et trouvée nuisible).

**Résultat : l'erreur d'estimation est divisée par 2** (RMSE 3,03 → 1,46 à r=2), **le classement ne
bouge pas** (86,56 % → 86,06 % @8). Le biais du coreset est un **mode commun** entre blocs : le
corriger ne change pas l'ordre.

### Concept 2 — Selection certifiée
La borne (*) étant bilatérale, elle permet une **séparation-évaluation** : éliminer par preuve tout
bloc dont la borne supérieure passe sous la k-ième borne inférieure. Ce serait le premier sélecteur
d'attention creuse avec **certificat de correction**.

**Résultat : échec total.** L'ensemble ambigu = 100 % des blocs, à tout rang, avec Hoeffding comme
avec Bernstein. Diagnostic : l'amplitude des écarts intra-cluster vaut `M = s||q||rho ≈ 23,9 nats`
en médiane. Le facteur de Bennett `g(M) = (e^M − 1 − M)/M²` vaut alors **4×10⁷** : toute inégalité
de concentration est vide. Les déviations ne sont pas petites — elles sont énormes.

### Concept 3 — Coreset calibré
Δ = l_b − lhat_b est **prévisible** : R² = 0,724 (RoPE) et 0,812 (NoPE) par régression sur deux
scalaires déjà stockés (rayon max, valeur propre max). J'ajuste les coefficients sur des couches
disjointes de celles de test.

**Résultat : dégrade légèrement** (80,63 % → 79,35 % @4). Le résidu vaut encore 1,36 nat.

### Concept 4 — Budget adaptatif
Puisque la masse est tantôt concentrée tantôt étalée, choisir k par requête pour atteindre une cible
de masse devrait battre un k fixe.

**Résultat : à budget moyen égal, le k fixe gagne légèrement** (46,33 % contre 43,45 % à k̄≈8).

---

## 4. La cause commune, et la vraie découverte

Les quatre échecs ont **une seule racine**, que voici mesurée :

| grandeur | avec RoPE | sans RoPE |
|---|---:|---:|
| écart de log-masse entre le 8ᵉ et le 9ᵉ bloc (médiane) | **0,070 nat** | **0,023 nat** |
| incertitude résiduelle du meilleur estimateur compact | 2,589 nats | 1,934 nats |
| **rapport** | **36,8×** | **85,3×** |

> **Les blocs sont quasi ex æquo.** Le signal à discriminer est 37 à 85 fois plus petit que
> l'incertitude de n'importe quel résumé compact. Aucun raffinement de l'estimateur — poids
> optimaux, certificats, calibration, adaptativité — ne peut franchir cet écart.

Trois conséquences que je crois importantes pour le domaine :

**(a) Le rappel d'ensemble est une mauvaise métrique.** Se tromper de bloc près de la coupure coûte
~0,07 nat, c'est-à-dire presque rien. C'est pourquoi le rappel de masse (85-90 %) est bien meilleur
que le rappel d'ensemble (60-75 %) — et c'est le rappel de masse qui compte. Optimiser le premier
est un faux objectif.

**(b) L'erreur dominante est la sparsité elle-même, pas le sélecteur.** Mesure de bout en bout de
l'erreur relative sur la **sortie d'attention** (k=8, blocs distants) :

| | erreur sortie |
|---|---:|
| coreset r=4 | 25,35 % |
| oracle (masse exacte) | 19,93 % |
| coreset r=4 à **k=16** | **16,85 %** |

Passer du coreset à un oracle parfait ne récupère que 5,4 points ; **doubler k en récupère 8,5**.
À budget réaliste, augmenter k rapporte plus que perfectionner le sélecteur.

**(c) La masse distante est diffuse, pas en aiguilles.** Il faut **61 blocs pour 90 % de la masse
distante** (sans RoPE), et seules 2,7 à 5,2 % des requêtes concentrent plus de la moitié de leur
masse distante sur un seul bloc. Le récit dominant — « il faut retrouver l'aiguille » — ne décrit
pas le régime observé sur ces modèles. C'est une queue diffuse qu'il faut capturer, et c'est
pourquoi le mean-pool reste une base étonnamment solide.

---

## 5. Trois propositions issues de ce travail

### A. Métrique — la distorsion de transformée de Laplace par octet
Le cadre du §1 donne un **étalon commun** à des techniques qu'on compare aujourd'hui mal :
compression latente (MLA), pooling (CSA), éviction, quantification sont *toutes* des façons de
perturber la mesure `mu_b`. Toutes devraient être évaluées par la même quantité — la distorsion
qu'elles infligent à `log integrale e^{sq.k} dmu` sur la loi des requêtes, rapportée à l'octet.
Cela permettrait enfin de dire si 1 octet de plus vaut mieux en centroïdes, en bits de
quantification ou en tokens conservés.

### B. Architecture — le coreset au service des trois branches
NSA et CSA maintiennent **trois branches** : compression, sélection, fenêtre glissante — donc deux
résumés distincts du même bloc. Un coreset hiérarchique unique peut servir les trois rôles :
- ses **centroïdes pondérés sont la représentation compressée** (on peut y porter l'attention
  directement : c'est la branche grossière, à la HCA) ;
- leur **log-sum-exp est le score de sélection** (§2) ;
- son **rayon est un signal d'incertitude gratuit** indiquant ce qu'on perd à ne pas lire le bloc.

Cela supprime une branche entière et ses paramètres. **[HYPOTHÈSE, non testée.]**

### C. Cible de recherche — déplacer l'effort du sélecteur vers k
Le §4(b) suggère que la communauté optimise le mauvais maillon. À erreur de sortie égale, il est
plus rentable de **rendre k plus grand mais moins cher** (blocs plus fins, KV mieux quantifié,
préchargement type LSA) que de raffiner le score de sélection. C'est cohérent avec mon analyse
antérieure ([SYNTHESE §3](../07_analyses/SYNTHESE.md)) : ce qui coûte, ce sont les **octets lus**,
et le sélecteur n'en gouverne qu'une petite part.

---

## 6. Reproduction

```bash
cd C:\Users\alici\Downloads\inference_opti\12_poc\code && python eval_selection.py --block 64
```

| Script | Rôle |
|---|---|
| [`capture_qk.py`](code/capture_qk.py) | capture des Q/K réels pré/post-RoPE |
| [`summaries.py`](code/summaries.py) | cadre théorique + tous les estimateurs |
| [`eval_selection.py`](code/eval_selection.py) | comparaison principale à budget égal |
| [`diagnostics.py`](code/diagnostics.py) | régime du développement (hypothèse réfutée) |
| [`diag_rank.py`](code/diag_rank.py) | spectre et rang effectif de la covariance |
| [`novel.py`](code/novel.py), [`certified.py`](code/certified.py) | EQS et certificats |
| [`diag_cert.py`](code/diag_cert.py) | **faisabilité des certificats — la mesure clé** |
| [`calibrated.py`](code/calibrated.py) | calibration + erreur de bout en bout |
| [`adaptive_k.py`](code/adaptive_k.py) | budget adaptatif |

---

## 7. Limites — à lire avant de citer ces résultats

1. **Modèles génériques, non entraînés pour le contexte long.** Qwen2.5-0.5B et SmolLM2-135M sur
   wikitext. COBS évalue un modèle **NSA entraîné avec son sélecteur dans la boucle** et affiné sur
   des données de type RULER. C'est une différence de fond : leur modèle a une attention bien plus
   *needle-like* que les miens. **Mes résultats portent sur l'estimation post-hoc de la masse de
   blocs dans un modèle dense pré-entraîné, pas sur un système entraîné bout en bout.** Il est
   parfaitement possible que le terme de covariance devienne utile quand le modèle s'adapte à lui.
2. **Petits modèles** (0,5 B et 135 M), contexte 8 k. Le régime à 128 k–1 M peut différer.
3. **Valeurs = clés.** N'ayant pas capturé V, l'erreur de sortie de §4(b) utilise K comme proxy de V.
   Les comparaisons relatives restent valides, les valeurs absolues sont indicatives.
4. **Pas de mesure de temps.** Aucun GPU sur la machine ; le « coût » est un comptage d'octets
   stockés, pas un temps mesuré. Le coreset demande en outre un log-sum-exp sur r termes au scoring
   (contre un produit scalaire pour COBS) : moins bien adapté au matériel, ce que ce POC ne chiffre pas.
5. **Le partitionnement est fait hors ligne** (k-centre, Gonzalez) au moment du prefill. Son coût
   n'est pas comptabilisé. k-centre et k-moyennes donnent des résultats quasi identiques, ce qui
   **n'a pas confirmé** ma préférence théorique pour le k-centre.

## 8. Ce qu'il faudrait faire ensuite

1. **Rejouer sur un modèle entraîné en creux** (DeepSeek-V3.2-Exp est public) et sur RULER. C'est
   la limite n°1 et la seule façon de trancher entre mon résultat et celui de COBS.
2. **Chiffrer le coût matériel du scoring log-sum-exp** contre la forme quadratique de COBS.
3. **Tester la proposition B** (coreset unique pour les trois branches) par entraînement.
4. **Mesurer l'écart d'ex æquo (§4) sur des modèles long-contexte** : s'il reste de l'ordre de
   0,07 nat, la conclusion « le sélecteur n'est pas le maillon faible » se généralise et devrait
   réorienter une bonne partie de l'effort du domaine.
