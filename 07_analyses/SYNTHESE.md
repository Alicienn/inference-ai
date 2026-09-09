# Synthèse — Optimisation de l'inférence IA

**Date :** 30 août 2026 · **Corpus :** 38 papiers arXiv (voir [INDEX.md](../INDEX.md))
**Méthode :** lecture intégrale des papiers structurants, reconstruction indépendante de leurs
chiffres à partir des seuls hyperparamètres publiés, puis expériences numériques sur activations
réelles (CPU uniquement — aucun GPU sur la machine).

Chaque affirmation est marquée **[FAIT]** (établi par un papier ou par une mesure reproductible),
**[INFÉRENCE]** (déduction de ma part à partir de faits) ou **[HYPOTHÈSE]** (spéculatif, non vérifié).

---

## 1. Le résultat central : ce qui limite l'inférence a changé de nature

Le domaine se raconte encore comme une lutte contre le coût quadratique de l'attention. Les
chiffres du corpus disent autre chose. En décodage à contexte long, **le calcul d'attention
proprement dit est devenu négligeable ; ce sont le *mécanisme de sélection* et la *bande passante
mémoire* qui dominent.**

Décomposition des FLOPs de décodage à 1M de tokens, reconstruite indépendamment (voir §2) :

| Modèle | FLOPs/token | dont GEMM | dont attention | dont **indexeur** |
|---|---:|---:|---:|---:|
| DeepSeek-V3.2 | 1 108 G | 74 G | 35 G | **999 G (90,2 %)** |
| DeepSeek-V4-Flash | 131 G | 26 G | 24 G | **82 G (62,3 %)** |
| DeepSeek-V4-Pro | 292 G | 98 G | 75 G | **119 G (40,7 %)** |

**[FAIT]** À 1M de contexte, 90 % du coût de décodage de DeepSeek-V3.2 est consommé par le
*lightning indexer* de DSA — c'est-à-dire par le mécanisme censé rendre l'attention économique.
L'attention creuse a déplacé le goulot d'étranglement plutôt que de le supprimer : sélectionner
`k` éléments parmi `n` coûte O(n), et cet O(n) finit par dominer le O(k) qu'il permet d'économiser.

**[INFÉRENCE]** C'est la clé de lecture de DeepSeek-V4. Son innovation n'est pas « encore plus de
sparsité » mais **compresser avant de sélectionner** : CSA agrège d'abord 4 tokens en une entrée,
puis n'indexe que `n/4` blocs. Le terme dominant est divisé par `m`. C'est un déplacement du
problème d'un cran en amont, et c'est ce qui explique que la part de l'indexeur retombe de 90 % à
62 %. La suite du papier (HCA, précisions basses) affine ; ce choix-là porte l'essentiel.

---

## 2. Vérification indépendante des chiffres de DeepSeek-V4

DeepSeek-V4 (arXiv 2606.19348, §1 et Fig. 1) annonce pour V4-Flash à 1M de contexte : **10 % des
FLOPs** (9,8× moins) et **7 % du KV cache** (13,7× moins) par rapport à V3.2 ; pour V4-Pro, 27 %
et 10 %. J'ai reconstruit ces quantités à partir des seuls hyperparamètres de §4.2.1, sans
utiliser les résultats annoncés.

**Validation préalable du modèle** — reproduire les comptes de paramètres publiés :

| Modèle | Total reconstruit | Publié | Écart | Activés reconstruits | Publié | Écart |
|---|---:|---:|---:|---:|---:|---:|
| V3.2 | 675,3 B | 671 B | +0,6 % | 40,9 B | 37 B | +10,5 % |
| V4-Flash | **284,2 B** | 284 B | **+0,1 %** | **13,2 B** | 13 B | **+1,4 %** |
| V4-Pro | 1 572,7 B | 1 600 B | −1,7 % | 48,6 B | 49 B | −0,9 % |

**[FAIT]** Le modèle architectural est validé : il retrouve 284B/13B pour V4-Flash à ~1 % près.
Les prédictions de coût qui en découlent sont donc fondées.

**Résultats de la vérification :**

| Grandeur (à 1M) | Annoncé | Reconstruit | Verdict |
|---|---:|---:|---|
| KV cache, V4-Flash vs V3.2 | 13,7× / 7 % | **14,2× / 7,0 %** | ✅ à 4 % |
| KV cache, V4-Pro vs V3.2 | 9,5× / 10 % | **9,8× / 10,2 %** | ✅ à 3 % |
| KV cache vs baseline BF16 GQA8 | « ~2 % » | **1,87 %** | ✅ |
| FLOPs, V4-Pro vs V3.2 | 3,7× / 27 % | **3,79× / 26,4 %** | ✅ à 2 % |
| FLOPs, V4-Flash vs V3.2 | 9,8× / 10 % | **8,43× / 11,9 %** | ⚠️ à 14 % |

**[FAIT]** Les chiffres de DeepSeek sont reproductibles à partir de leur seule description
architecturale. C'est un point de crédibilité important : le papier est vérifiable.

**[INFÉRENCE]** Le seul écart notable (FLOPs de V4-Flash, 8,43× contre 9,8×) provient de ma
reconstruction de la *baseline* V3.2, que j'estime à 1,11 TFLOP/token quand la Figure 1 la situe
vers 1,2 T. En prenant leur valeur de baseline, le rapport devient 9,1× — soit un accord à 7 %.
L'écart résiduel tient vraisemblablement au nombre de têtes de l'indexeur de V3.2, que le papier
V3.2 ne précise pas explicitement (j'ai retenu 64×128, la configuration publique de V3.2-Exp).

Les conditions exactes de reproduction sont dans [`METHODES.md`](METHODES.md) ; le code est dans
[`_tools/model_inference_cost.py`](../_tools/model_inference_cost.py).

---

## 3. La distinction que le domaine ne fait pas : cache *résident* vs cache *lu*

Tous les papiers de compression du KV cache rapportent une réduction de **taille résidente**
(l'empreinte mémoire). C'est la bonne métrique pour la capacité — combien de séquences tiennent en
HBM. **Ce n'est pas la métrique de la latence.** Ce qui fixe la latence, c'est le nombre d'octets
réellement *lus* à chaque token décodé.

Pour l'attention dense, les deux coïncident. **Pour l'attention creuse, elles divergent — et
l'écart est systématiquement en faveur de la sparsité.**

| À 1M de contexte | KV résident | KV lu/token | ratio |
|---|---:|---:|---:|
| DeepSeek-V3.2 | 43,6 GB | 7 888 MB | 5,9× |
| DeepSeek-V4-Flash | 3,07 GB | 423 MB | 7,8× |
| DeepSeek-V4-Pro | 4,46 GB | 630 MB | 7,6× |

**[FAIT]** Le gain de V4-Flash sur V3.2 mesuré en *octets lus par token* est de **18,6×**, contre
13,7× mesuré en taille résidente. L'avantage de latence dépasse donc l'avantage mémoire annoncé
d'un facteur ~1,4.

**[INFÉRENCE]** Les papiers de compression KV sous-déclarent leur propre bénéfice en latence,
parce qu'ils rapportent la métrique de capacité. Je recommande de rapporter systématiquement les
deux. Une méthode d'éviction qui réduit la taille résidente sans réduire les octets lus (parce
qu'elle doit scanner tout le cache pour décider quoi évincer) n'apporte aucun gain de latence —
et ce cas est fréquent.

**[FAIT]** L'intensité arithmétique de l'attention en décodage vaut 131 à 308 FLOP/octet, contre
un point de bascule de 563 (B200) à 591 (H800) FLOP/octet. Le décodage est donc **toujours limité
par la bande passante**, jamais par le calcul — y compris pour V4-Pro. Optimiser les FLOPs
d'attention en décodage sans réduire les octets lus est sans effet.

---

## 4. Pourquoi la compression par blocs fonctionne (et mon hypothèse initiale était fausse)

J'ai testé le pari central de V4 sur des cartes d'attention **réelles** (GPT-2, 12 couches ×
12 têtes, textes wikitext), en comparant la masse d'attention récupérée par sélection top-k au
niveau token (DSA) et au niveau bloc (CSA).

**Hypothèse initiale — infirmée.** Je supposais que la compression par blocs marche parce que
l'attention est spatialement groupée (des tokens voisins seraient pertinents ensemble). Les
mesures disent le contraire : l'autocorrélation spatiale de la masse d'attention tombe à **0,207
dès le décalage 1** et 0,13 au décalage 4 ; **30,4 %** des tokens du top-16 sont des « aiguilles »
isolées sans voisin pertinent. L'attention n'est pas groupée. **[FAIT]**

**Le mécanisme réel.** La sélection par blocs conserve pourtant 93,7 % à 99,3 % de la masse à
m=4. L'explication est différente : scorer un bloc par la *somme* de ses tokens préserve la
détectabilité d'une aiguille — un bloc contenant une aiguille reste bien classé. Le coût de la
granularité n'est donc **pas une perte de détection mais une dilution du budget** : on lit m
tokens pour n'en exploiter qu'un. **[INFÉRENCE]**

C'est ce qui rend l'arbitrage favorable, car les deux termes n'ont pas la même échelle :

- **Gain** : le coût de l'indexeur passe de `n` à `n/m` clés — terme qui *croît avec le contexte*
  et domine tout le reste (§1).
- **Perte** : la dilution ne touche que l'attention centrale, en O(top-k) — terme **constant**,
  borné, et petit.

**[FAIT]** À bande passante d'indexeur égale, la sélection par blocs domine nettement :

| budget DSA | DSA (m=1) | CSA m=2 | CSA m=4 | CSA m=8 | CSA m=16 |
|---:|---:|---:|---:|---:|---:|
| 32 | 0,8080 | 0,8468 (+4,8 %) | 0,8870 (+9,8 %) | 0,9295 (+15,0 %) | 0,9750 (+20,7 %) |
| 64 | 0,8664 | 0,9032 (+4,2 %) | 0,9410 (+8,6 %) | 0,9802 (+13,1 %) | — |
| 128 | 0,9179 | 0,9516 (+3,7 %) | 0,9844 (+7,2 %) | — | — |

**[INFÉRENCE]** Ce tableau justifie CSA, mais suggère aussi que **m=4 est conservateur** : m=8 et
m=16 font mieux à budget égal dans mon protocole. DeepSeek a probablement choisi m=4 pour des
raisons que ma mesure ne capture pas — qualité en tâches de récupération fine, stabilité
d'entraînement, ou interaction avec le `top-k=512`. C'est une question ouverte (voir
[QUESTIONS_OUVERTES.md](QUESTIONS_OUVERTES.md), Q1).

---

## 5. Une faiblesse structurelle de CSA que le papier ne discute pas : le trou de Jensen

Mon analyse §4 suppose que le bloc est scoré par la **somme** des attentions de ses tokens. Or CSA
ne fait pas cela : il score par produit scalaire avec une **clé compressée**, moyenne pondérée
apprise des clés du bloc (éq. 11-12 du papier). Par l'inégalité de Jensen :

```
exp( moyenne_j(logit_j) )  ≤  moyenne_j( exp(logit_j) )
```

La mise en commun des clés **sous-estime systématiquement** un bloc contenant une aiguille. J'ai
quantifié ce trou en faisant varier la « piquosité » β de la pondération de compression (β=0 :
moyenne uniforme ; β→∞ : max).

**[FAIT]** Masse d'attention récupérée, budget 256, attention réelle :

| m | β=0 (moyenne) | β=0,5 | β→∞ (max) | oracle somme | trou de Jensen |
|---:|---:|---:|---:|---:|---:|
| 4 | 0,8127 | **0,9409** | 0,9391 | 0,9410 | **+15,6 %** |
| 8 | 0,7599 | **0,9294** | 0,9265 | 0,9295 | **+21,9 %** |
| 16 | 0,7170 | **0,9164** | 0,9127 | 0,9166 | **+27,3 %** |

Trois enseignements :

1. **[FAIT]** Un compresseur à moyenne uniforme perd 13,6 % (m=4) à 21,8 % (m=16) de la masse par
   rapport à l'oracle. Le risque est réel et **croît avec le taux de compression** — ce qui
   fournit une explication possible du choix conservateur m=4.
2. **[FAIT]** Le trou se referme dès β≈0,25–0,5, c'est-à-dire dès que la pondération devient
   *légèrement* piquée. Il n'est pas nécessaire que le compresseur apprenne un max.
3. **[FAIT]** Le maximum pur (β→∞) est **légèrement moins bon** que β≈0,5 : il ignore les blocs
   portant plusieurs tokens moyens. L'optimum est intérieur.

**[INFÉRENCE]** Conséquence de conception non explicitée dans le papier V4 : la qualité de CSA ne
dépend pas seulement de `m`, mais de **l'entropie de la pondération de compression**
`Softmax_row` (éq. 11). Un compresseur à softmax plat annule une part importante du bénéfice de la
compression. Cette entropie devrait être **surveillée comme métrique d'entraînement**, et son
initialisation (via les biais positionnels apprenables `B_a, B_b`) traitée comme un
hyperparamètre critique. À ma connaissance, aucun papier du corpus ne le mentionne.

---

## 6. Compression du KV cache : l'allocation de bits domine le choix du format

Le papier RoPE-Aware Bit Allocation (arXiv 2606.24033) pose que le logit d'attention RoPE se
décompose en une somme sur des blocs de fréquence 2D, dont l'énergie serait « fortement inégale ».
J'ai testé cette prémisse sur les activations réelles de SmolLM2-135M (30 couches, GQA, RoPE).

**[FAIT] Prémisse confirmée, et amplement :** sur 90 profils (couche × tête KV), l'étendue
max/médiane de l'énergie par bloc vaut **793× en médiane** (p90 : 2 688×, max : 4 873×), soit
près de 3 ordres de grandeur.

**[FAIT] Structure supplémentaire non soulignée par le papier :** l'énergie n'est pas répartie au
hasard entre fréquences. Les blocs **basse fréquence** (rotation lente, i≥16) portent **34,9×**
plus d'énergie que les blocs haute fréquence, avec une concentration marquée sur les blocs 22-25
(jusqu'à 528× la médiane).

**[FAIT] L'allocation guidée par l'énergie bat nettement l'uniforme**, à budget de bits moyen égal
(quantification appliquée après RoPE, comme dans les systèmes réels) :

| bits moyens | erreur logit uniforme | erreur allouée | gain | KL uniforme | KL alloué |
|---:|---:|---:|---:|---:|---:|
| 2 | 0,2397 | **0,0424** | 82,3 % | 0,8375 | **0,0225** |
| 3 | 0,0798 | **0,0212** | 73,5 % | 0,1440 | **0,0054** |
| 4 | 0,0341 | **0,0088** | 74,2 % | 0,0315 | **0,0010** |
| 6 | 0,0077 | **0,0018** | 76,3 % | 0,0018 | **0,00004** |

**[INFÉRENCE]** Traduit en budget : l'allocation guidée vaut environ **1,4 à 2 bits gratuits**.
Un cache à 3 bits alloués est de meilleure qualité qu'un cache à 4 bits uniformes. C'est un gain
de 25 à 40 % de mémoire à qualité constante, sans changer de format numérique — bien supérieur à
ce que rapporte le passage d'un format à un autre.

**[INFÉRENCE] Lecture critique du choix de DeepSeek-V4.** V4 stocke les 64 dimensions RoPE en
BF16 et le reste en FP8 (§2.3.4) : c'est une allocation **binaire**, un cas très grossier de
l'allocation par blocs. Mes mesures montrent que l'écart d'énergie décisif se joue *entre bandes
de fréquence à l'intérieur même de la partie RoPE* — gradient qu'une allocation à deux niveaux ne
peut pas capturer. Il y a là un gain résiduel plausible pour V4, et une convergence naturelle
entre la littérature « architecture » (DeepSeek) et la littérature « quantification »
(Block-GTQ, TurboQuant) qui ne se citent pas.

---

## 7. Résultat original : décodage spéculatif et attention creuse se nuisent

Ce point ne figure, à ma connaissance, dans aucun papier du corpus, alors que DeepSeek-V4 combine
précisément les deux techniques (MTP depth 1 + CSA).

**Le raisonnement.** En contexte long, le décodage est limité par la lecture du KV cache (§3). Le
décodage spéculatif vérifie γ+1 candidats en une passe : en attention **dense**, le cache n'est lu
qu'**une seule fois** pour tous les candidats. La spéculation y est donc bien plus rentable qu'on
ne le dit d'ordinaire, car elle amortit le terme dominant. **[INFÉRENCE]**

**Le problème.** En attention **creuse**, chaque position candidate sélectionne son *propre* top-k.
La vérification doit lire l'**union** des sélections. Si celles-ci divergent, l'amortissement
s'effondre. J'ai mesuré ce recouvrement sur attention réelle (top-64) :

**[FAIT]** Efficacité d'amortissement de la lecture du KV :

| γ | dense | creux m=1 (DSA) | creux m=4 (CSA) |
|---:|---:|---:|---:|
| 1 | 100 % | 76,8 % | **87,3 %** |
| 2 | 100 % | 65,3 % | **80,5 %** |
| 4 | 100 % | 53,5 % | **72,8 %** |
| 8 | 100 % | 41,8 % | **65,2 %** |

**[FAIT]** Accélération résultante (modèle borné par la bande passante, tête de brouillon type
EAGLE-3 à coût 0,1) :

| α | γ optimal dense | accél. dense | accél. creux m=1 | accél. creux m=4 |
|---:|---:|---:|---:|---:|
| 0,7 | 4 | 1,98× | 1,26× (γ=2) | 1,57× (γ=3) |
| 0,8 | 6 | 2,47× | 1,48× (γ=4) | 1,92× (γ=6) |
| 0,9 | 8 | 3,40× | 1,92× (γ=8) | 2,63× (γ=8) |

Trois conséquences :

1. **[FAIT]** L'attention creuse par token ampute sévèrement le décodage spéculatif : à α=0,7 et
   γ=8, l'accélération tombe de 1,78× à **1,00×** — la spéculation ne rapporte plus rien.
2. **[FAIT]** L'attention creuse **abaisse le γ optimal**. Les réglages de γ publiés pour des
   modèles denses ne se transposent pas.
3. **[INFÉRENCE] La granularité par blocs sauve partiellement la spéculation** : m=4 récupère
   72,8 % d'amortissement contre 53,5 % pour m=1 (γ=4), parce que des requêtes voisines
   sélectionnent plus souvent les *mêmes blocs* que les mêmes tokens. **C'est un argument
   supplémentaire en faveur de CSA, que DeepSeek ne formule pas** : leur choix de blocs rend leur
   propre MTP viable.

---

## 8. Serving désagrégé : une hypothèse intuitive, et fausse

**[HYPOTHÈSE initiale, INFIRMÉE]** J'ai supposé que la taille du KV cache interdisait la
désagrégation prefill/decode en contexte long, le transfert inter-nœud devenant prohibitif.

**[FAIT]** C'est faux à prefill froid. À 1M, V3.2 transfère 43,6 GB, soit 0,94 s sur InfiniBand
400G — mais le prefill lui-même coûte ~49 s. Le transfert ne pèse que **1,9 %**. Et la part
*décroît* avec le contexte (9,9 % à 8K, 1,9 % à 1M) puisque le prefill croît plus vite que le
cache. Le calcul domine partout.

**[FAIT] Le régime où le transfert devient décisif est la réutilisation de préfixe.** C'est
précisément la thèse de Mooncake (2407.00079), architecture « KVCache-centric » dont le gain vient
des hits de cache (multi-tours, agents, RAG). En cas de hit, il n'y a plus de prefill à amortir :
le transfert est 100 % du coût.

| Servir un préfixe de 1M déjà en cache | NVLink | IB 400G | Eth 100G |
|---|---:|---:|---:|
| DeepSeek-V3.2 | 52 ms | **937 ms** | 3 748 ms |
| DeepSeek-V4-Flash | 4 ms | **66 ms** | 264 ms |

**[INFÉRENCE]** C'est là que la compression architecturale du KV change la nature du système :
avec V3.2, un hit de cache à 1M coûte ~0,9 s de transfert inter-nœud, ce qui annule presque
l'intérêt du cache et force à confiner la désagrégation au domaine NVLink. Avec V4-Flash, la
réutilisation de préfixe redevient franchement rentable sur réseau standard. **Un choix de niveau
modèle débloque une technique de niveau système** — les deux littératures se lisent mal séparément.

---

## 9. Validation croisée sur une seconde architecture

La limite la plus sérieuse des §4, §5 et §7 était leur dépendance à **GPT-2** (117 M, positions
apprises, attention dense classique). J'ai répliqué les résultats structurels sur **SmolLM2-135M**
— architecture moderne, famille et entraînement différents (RoPE, GQA 9/3, 30 couches, SwiGLU).

**[FAIT] Les quatre résultats se répliquent, et le plus souvent plus nettement :**

| Mesure | GPT-2 | SmolLM2-135M | Verdict |
|---|---:|---:|---|
| Masse dans le top-64 | 0,876 | **0,899** | ✅ concentration confirmée, plus forte |
| Masse dans le top-1 % des positions | 0,655 | **0,722** | ✅ plus forte |
| Autocorrélation au décalage 1 | 0,207 | **0,129** | ✅ regroupement spatial **encore plus faible** |
| Aiguilles isolées (top-16) | 30,4 % | **46,6 %** | ✅ encore plus d'aiguilles |
| Masse conservée à m=4 (budget 256) | 97,9 % | **97,9 %** | ✅ identique |
| Gain à bande passante d'indexeur égale (m=4) | +9,8 % | **+6,1 %** | ⚠️ même sens, plus faible |
| Amortissement spéculatif, m=1, γ=4 | 53,5 % | **45,2 %** | ✅ interaction négative **plus forte** |
| Amortissement spéculatif, m=4, γ=4 | 72,8 % | **62,3 %** | ✅ sauvetage par blocs confirmé |

**[INFÉRENCE]** Trois conséquences pour la lecture de ce rapport :

1. **Le §4 est renforcé.** Le regroupement spatial est encore plus faible sur SmolLM2 (ρ=0,129) et
   les aiguilles isolées plus nombreuses (46,6 %), alors même que la sélection par blocs conserve
   la même masse. L'explication par « préservation de la détectabilité, dilution du budget » sort
   confirmée ; l'explication par regroupement spatial est définitivement écartée.
2. **Le §5 devient plus urgent.** Avec 46,6 % d'aiguilles isolées, le trou de Jensen — qui pénalise
   précisément les blocs à aiguille — concerne une part plus grande des sélections sur les
   architectures modernes que sur GPT-2.
3. **Le §7 est renforcé et aggravé.** L'amortissement spéculatif tombe à 33,7 % (m=1, γ=8) contre
   41,8 % sur GPT-2. L'interaction négative entre spéculation et sparsité est donc **plus sévère**
   sur architecture moderne, et le bénéfice de la granularité par blocs (+17 points à γ=4) se
   confirme.

**[FAIT] Seule nuance :** le gain à bande passante d'indexeur égale est plus modeste sur SmolLM2
(+6,1 % contre +9,8 % à m=4). Le sens et le classement sont préservés, l'amplitude non. Les
amplitudes de ce rapport doivent donc être lues comme des **ordres de grandeur dépendants du
modèle**, pas comme des constantes.

Code : [`_tools/exp_E_crossvalidation.py`](../_tools/exp_E_crossvalidation.py).

---

## 10. Ce qu'il faut retenir

1. **[FAIT]** Le goulot d'étranglement du décodage long-contexte n'est ni l'attention ni les FLOPs,
   mais **le mécanisme de sélection et la bande passante**. 90 % du coût de V3.2 à 1M est son
   propre indexeur.
2. **[INFÉRENCE]** L'idée directrice de V4 — *compresser avant de sélectionner* — attaque
   exactement ce terme. C'est la bonne cible, et les chiffres annoncés sont vérifiables (§2).
3. **[FAIT]** Il faut distinguer **KV résident** et **KV lu par token**. La seconde métrique, seule
   pertinente pour la latence, est absente de la littérature et favorise la sparsité (18,6× contre
   13,7× annoncés).
4. **[INFÉRENCE]** La compression par blocs marche par **préservation de la détectabilité**, pas
   par regroupement spatial — lequel n'existe pas dans les données (ρ=0,21).
5. **[INFÉRENCE]** Le point faible de CSA est **l'entropie du compresseur** (trou de Jensen,
   jusqu'à −21,8 %), non le taux de compression. Facile à corriger, non discuté.
6. **[INFÉRENCE]** En quantification du KV, **l'allocation de bits vaut ~2 bits** — plus que le
   choix du format. Le gradient d'énergie inter-fréquences (35×) est inexploité par l'allocation
   binaire de V4.
7. **[INFÉRENCE, original]** **Attention creuse et décodage spéculatif se nuisent** (amortissement
   à 41,8 %) ; la granularité par blocs les réconcilie partiellement (65,2 %).
8. **[FAIT]** L'intuition « le transfert KV interdit la désagrégation » est **fausse** à prefill
   froid, et **vraie** sous réutilisation de préfixe.

---

## Documents liés

- [`axe1_attention_creuse.md`](axe1_attention_creuse.md) — lignée NSA → DSA → CSA/HCA → LSA
- [`axe2_kv_cache.md`](axe2_kv_cache.md) — quantification, éviction, allocation de bits
- [`axe3_decodage_speculatif.md`](axe3_decodage_speculatif.md) — EAGLE-3 et son interaction avec la sparsité
- [`axe4_serving.md`](axe4_serving.md) — désagrégation, scheduling, économie
- [`QUESTIONS_OUVERTES.md`](QUESTIONS_OUVERTES.md) — 10 pistes de recherche classées
- [`LACUNES_CORPUS.md`](LACUNES_CORPUS.md) — ce qui manque au corpus
- [`METHODES.md`](METHODES.md) — reproduction, hypothèses, limites

## Addendum v2 — ce que la re-vérification à la source a changé

**Corrections d’erreurs v1** (plus aucun chiffre non sourcé ne subsiste) :

1. La recette « 943,7B tokens sur 15 000 steps » appartient à **V3.2/DSA** [2512.02556], pas à V4.
2. « 576 dims ≈ 70 Ko/token » : 512 (c_KV) + 64 (k_R) dims/token/couche est un fait MLA [2412.19437] ; le total 576 et les ≈ 68,6 Kio/token sur 61 couches sont **notre arithmétique** [INFÉRENCE].
3. Le système PD d’Ascend (Huawei) est **TaiChi** [2508.01989] (la v1 l’appelait « PD-Pai »).
4. Le « 1,9×–3,6× » d’EAGLE-3 en v1 lisait des ticks d’axes de figure ; les chiffres vérifiés sont : 5,51× (harnais bs1) ; 158,34→373,25 tok/s sous SGLang bs1 (rapport ≈ 2,36×, **calculé**) ; 1,38× à b=64 ; 0,88× (EAGLE-1, b=48) ; 1,01× (EAGLE-3, b=56) [2503.01840].

**Faits consolidés nouveaux :**

- **TaiChi** [2508.01989] : aucun régime (agrégé/désagrégé) n’est universellement optimal ; bascule de phase unifiée — goodput +77 %, TTFT ÷13,2, TPOT ÷1,69.
- **Mistletoe** [2605.14005] : un suffixe discret effondre la longueur acceptée τ (EAGLE 5,47×→1,83× sur GSM8K ; MT-bench ÷1,89) sans dégradation visible de qualité — surface DoS économique du spéculatif multi-tenant.
- **LSA n’est pas gratuite** [2606.09079, §3.3.2] : MRCR 76,0→48,0 % ; fuite volumique du gater (rétention 8,4 %, volume retenu ×2,5).
- **Mooncake** [2407.00079] : +525 % de throughput agrégé en production, +75 % de requêtes servies sous contraintes SLO.

**Corpus** : passé de 31 à 38 PDFs en cours de session (Mistletoe [2605.14005] ajouté, texte extrait et vérifié). Ajouts restant à extraire : voir `LACUNES_CORPUS.md`. Méthodologie : `METHODES.md`.
