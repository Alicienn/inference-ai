# Journal de recherche — optimisation de l'inférence

Journal brut : raisonnements en cours, dérivations, pistes abandonnées et pourquoi.
Conventions : [FAIT] source · [DÉRIVATION] calcul refait par moi · [INFÉRENCE] déduction ·
[HYPOTHÈSE] proposition non vérifiée · [ABANDON] piste creusée puis écartée, avec la raison.

---

## Session 3 — 30 août 2026

### Point de départ
Acquis des sessions précédentes (voir `07_analyses/` et `12_poc/`) :
- vérification indépendante des coûts DeepSeek-V4 (±1,7 % sur les paramètres) ;
- le décodage est borné par la bande passante (intensité 131-308 FLOP/octet vs bascule ~563) ;
- l'indexeur consomme 90 % des FLOPs de V3.2 à 1M ;
- POC : coreset > covariance pour les résumés de blocs (4,5× moins cher que COBS) ;
- **découverte centrale non exploitée** : les blocs sont quasi ex æquo (écart 0,070 nat entre
  le 8e et le 9e) alors que l'incertitude d'estimation vaut 1,4-2,6 nats.

Ce dernier point est resté au stade du constat. C'est là qu'est la profondeur : il faut en
faire une **théorie prédictive**, pas une observation.

### Question directrice de la session
> Existe-t-il une grandeur scalaire unique qui résume la qualité d'un sélecteur de blocs,
> et dont on puisse dériver une borne supérieure de performance atteignable à budget donné ?

Si oui, on cesse de comparer des méthodes une à une : on compare des points sur une courbe,
et on sait ce qui reste à gagner.

### Première dérivation : quelle est la bonne mesure d'erreur ?

[DÉRIVATION] Observation qui déclenche tout. Dans le POC, EQS divise le RMSE par 2
(3,03 → 1,46) **sans améliorer le rappel**. Donc le RMSE n'est PAS la bonne variable.

Décomposons l'erreur d'un estimateur pour une requête q donnée :
    e_b = lhat_b(q) - l_b(q)  =  m(q)  +  eps_b(q)
avec m(q) = moyenne de e_b sur les blocs (mode commun, propre à la requête)
et eps_b(q) l'écart au mode commun.

Le classement des blocs ne dépend QUE de eps_b : ajouter une constante m(q) à tous les
scores ne change aucun ordre. Donc :
    la variable pertinente est  sigma_disc = ecart-type_b( eps_b(q) ),
    et non le RMSE qui mélange m(q) et eps_b.

C'est exactement ce que EQS a corrigé : le mode commun. D'où l'absence de gain.

[HYPOTHÈSE à tester immédiatement] **Loi universelle de sélection** : le rappel de masse
d'un sélecteur ne dépend de lui que par sigma_disc. Deux méthodes de sigma_disc égal
donnent le même rappel, quelle que soit leur construction.

Si cette loi tient, elle a trois conséquences fortes :
1. on peut classer toutes les méthodes sur un seul axe ;
2. on peut prédire le rappel d'une méthode future à partir de son seul sigma_disc ;
3. couplée à une borne débit-distorsion sur sigma_disc(budget), elle donne un
   **plafond théorique de rappel à budget d'octets donné** — donc ce qui reste à gagner.

Test en cours dans `12_poc/code/law_selection.py`.

### Résultat 1 : la loi de sélection tient à moitié, et son échec est instructif

[DÉRIVATION] Courbe de référence obtenue par injection de bruit gaussien contrôlé sur les
log-masses EXACTES (Qwen2.5-0.5B, L=64, blocs distants). Elle donne le rappel atteignable
en fonction du seul niveau de bruit, indépendamment de toute méthode :

| sigma (nats) | rappel@4 | rappel@8 | rappel@16 |
|---:|---:|---:|---:|
| 0,05 | 99,3 % | 99,4 % | 99,6 % |
| 0,10 | 97,7 % | 98,1 % | 98,6 % |
| 0,20 | 93,3 % | 94,4 % | 95,8 % |
| 0,50 | 79,7 % | 82,7 % | 86,4 % |
| 1,00 | 66,1 % | 70,1 % | 75,4 % |
| 2,00 | 53,4 % | 58,1 % | 64,7 % |

[FAIT] Les méthodes réelles ne tombent PAS toutes dessus. Structure des résidus (rappel@8) :
- coreset r=2/4/8 et mean-pool : écart **+0,04 à +0,84 point** — la loi les prédit
  quasi parfaitement ;
- COBS r=1..8 et GMM : écart **+1,7 à +8,6 points** — ils font bien MIEUX que prédit ;
- maxpool et Quest : écart **−1,4 à −7,5 points** — bien PIRE que prédit.

[ABANDON PARTIEL] J'ai testé le raffinement « seule la composante de l'erreur ORTHOGONALE au
signal nuit au classement » (une erreur proportionnelle au signal ne fait que le
redimensionner, sans changer l'ordre). Décomposition eps_b = a(l_b - lbar) + eps_perp.
Résultat : gain marginal (écart 4,32 → 3,88 points) et les pentes a mesurées sont toutes
quasi nulles (−0,27 à +0,15). **Ce n'est donc pas la colinéarité qui explique l'écart.**

[INFÉRENCE] L'explication restante est l'HÉTÉROSCÉDASTICITÉ. Le bruit de référence est
homoscédastique et indépendant de la masse ; celui de COBS ne l'est pas : son terme
quadratique est toujours positif et croît avec la dispersion du bloc, or les blocs de forte
masse sont aussi ceux de forte dispersion. Son erreur est donc concentrée sur des blocs qui
seraient bien classés de toute façon, et son bruit EFFECTIF à la frontière de décision est
plus petit que son sigma global. Symétriquement, Quest a une erreur énorme et mal placée.

[INFÉRENCE] Conclusion honnête : **sigma seul ne suffit pas ; c'est sigma DANS LA ZONE DE
CONTENTION (autour du rang k) qui gouverne.** La loi reste utile comme borne et comme
diagnostic, pas comme prédicteur exact.

### Ce que la courbe de référence apporte quand même — et c'est l'essentiel

[DÉRIVATION] Elle chiffre la MARGE RESTANTE, ce qu'aucun papier du corpus ne fait :
- pour 95 % de rappel@8 il faut sigma <= 0,20 nat ;
- pour 99 % il faut sigma <= 0,05 nat ;
- le meilleur résumé compact mesuré (coreset r=8) est à sigma_disc = 0,30 (sans RoPE)
  et 0,63 (avec RoPE).

Croisé avec l'écart d'ex aequo mesuré (0,070 nat entre le 8e et le 9e bloc), cela borne
tout le domaine utile : **sigma pertinent ∈ [0,05 ; 1] nat**. En dessous de 0,05 on ne gagne
plus rien (on est déjà à 99 %) ; au-dessus de 1 on est au niveau du mean-pool.

### Changement de piste : et si le problème n'était pas l'estimateur mais l'ALLOCATION ?

Toutes les méthodes du corpus (NSA, Quest, COBS, CSA, et mes coresets) partagent une
**hypothèse implicite non questionnée** : *le même budget d'octets est alloué à chaque bloc,
et chaque bloc est lu exactement une fois.* C'est un schéma à passe unique et à résolution
uniforme.

Or le problème « trouver les k plus grands parmi n, à partir d'estimations bruitées, sous
budget de mesure » est un problème classique et RÉSOLU dans un autre domaine : celui des
**bandits stochastiques**, sous le nom d'identification des meilleurs bras. Sa théorie dit
deux choses que la littérature de l'attention creuse ignore :
1. l'allocation optimale du budget est **séquentielle et non uniforme** (élimination
   successive : on dépense sur les prétendants, pas sur les éliminés) ;
2. il faut distinguer l'IDENTIFICATION (retrouver le vrai top-k, coûteuse en 1/Delta^2)
   du REGRET SIMPLE (perdre peu de masse), bien plus facile — ce qui est exactement la
   distinction empirique rappel d'ensemble vs rappel de masse que j'ai mesurée.

C'est la piste à creuser. Je vais chercher les sources primaires de ce domaine.

### Recherche d'antériorité — et ce qu'elle a appris

Avant d'implémenter, j'ai cherché si l'idée existait. Elle existe en partie :
- [FAIT] **AsyncTLS** (2604.07815, Meituan) : hiérarchie à deux niveaux, mais elle raffine la
  GRANULARITÉ (filtrage de blocs grossier -> sélection de tokens fine), à résumé fixe.
- [FAIT] **HiSparse** (2608.07009) : hiérarchie de STOCKAGE (historique en mémoire hôte,
  cache GPU borné), indexeur-agnostique.
- [FAIT] **HieraSparse** (2604.16864), **SALS** (2510.24273), **SeerAttention-R** (2506.08889).

[INFÉRENCE] Aucune ne raffine la **précision du résumé à granularité constante**, avec une
allocation du budget pilotée par la distance au seuil de décision. C'est l'espace laissé libre.
Côté théorie j'ai récupéré les sources primaires : Bubeck-Munos-Stoltz (0802.2655, regret
simple vs identification), Carpentier-Locatelli (1605.09004, borne inférieure serrée en budget
fixe), Jamieson lil'UCB (1312.7308). Nouveaux dossiers `14_hierarchique/` et `15_theorie_selection/`.

### Résultat 2 (POSITIF) : ASP — Allocation Séquentielle de Précision

[DÉRIVATION] Courbe débit-distorsion mesurée des résumés (Qwen, L=64) :

| coreset r | vec/bloc | sigma_disc | rappel@8 |
|---:|---:|---:|---:|
| 1 | 1,02 | 0,480 | 87,73 % |
| 2 | 2,03 | 0,408 | 89,14 % |
| 4 | 4,07 | 0,356 | 91,02 % |
| 8 | 8,14 | 0,302 | 92,62 % |
| 16 | 16,28 | 0,236 | 95,20 % |
| 32 | 32,6 | — | 97,81 % |

[FAIT] Gain d'ASP mesuré **honnêtement**, par interpolation du front plat (on cherche
combien d'octets il faudrait à un schéma uniforme pour atteindre EXACTEMENT le même rappel) :

| banc | gain médian | meilleur | configs perdantes |
|---|---:|---:|---:|
| Qwen, RoPE, passe 1 = r=2 | **1,64×** | 2,38× | 1/32 |
| Qwen, sans RoPE, passe 1 = r=1 | 1,55× | 2,25× | 2/40 |
| SmolLM2, RoPE, passe 1 = r=4 | 1,36× | 1,87× | 1/24 |
| SmolLM2, sans RoPE, passe 1 = r=4 | 1,29× | 1,66× | 2/24 |

[AUTOCORRECTION] Mon premier calcul annonçait « 4,06× ». C'était un **artefact de placement
de seuil** : je comparais au premier point plat dépassant 89,5 %, or ce point était très
au-dessus de la cible. L'interpolation donne le chiffre correct, entre 1,1× et 2,4×.
Je consigne l'erreur : elle illustre pourquoi il faut comparer à rappel égal interpolé et
jamais à seuil discret.

[FAIT] Ablation du filtre de passe 1 : un filtre plus riche (r=2 ou r=4 au lieu du mean-pool)
n'améliore guère le gain médian mais rend ASP **robuste** — les configurations perdantes
tombent de 15/40 à 2/24. **Recommandation pratique : passe 1 = coreset r=2.**

### Ce que ce gain vaut vraiment, ramené au système

[DÉRIVATION] En réutilisant le modèle roofline de la session 1, part de l'indexeur dans les
octets lus par token décodé :

| modèle | contexte | total lu/token | dont indexeur | part | gain total si indexeur ÷1,5 |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V3.2 | 1M | 7 888 MB | 7 808 MB | **99,0 %** | **1,49×** |
| DeepSeek-V4-Flash | 1M | 423 MB | 320 MB | **75,6 %** | **1,34×** |
| DeepSeek-V4-Pro | 1M | 630 MB | 464 MB | 73,7 % | 1,33× |
| DeepSeek-V4-Flash | 128k | 62 MB | 41 MB | 66,1 % | 1,28× |

[INFÉRENCE] ASP attaque donc le terme **dominant** du décodage long-contexte, pas un terme
secondaire. 1,5× sur le sélecteur ≈ 1,34× sur la latence totale à 1M. La boucle est fermée
avec le résultat de la session 1 (« l'indexeur consomme 90 % des FLOPs de V3.2 ») : c'est le
même goulot, vu côté octets.

[LIMITE] ASP échange du STOCKAGE contre de la BANDE PASSANTE : il faut garder en mémoire les
résumés fins de tous les blocs, on se contente de ne pas les lire. Comme le décodage est
borné par la bande passante et non par la capacité (établi session 1), l'échange va dans le
bon sens — mais il n'est pas gratuit : ~15-25 % de surcoût mémoire pour des résumés r=16.

### Sur les méthodes de découverte, et laquelle j'ai appliquée

En reconstituant les lignées, je repère six schémas de déblocage réutilisables :

| Avancée | Reformulation qui débloque | Méthode réutilisable |
|---|---|---|
| MLA (V2/V3) | q.k = c_Q^T (W_UQ^T W_UK) c_KV : la up-projection s'absorbe dans la requête, on ne matérialise jamais K | **exploiter l'associativité pour déplacer du calcul à travers la frontière de stockage** |
| DSA -> CSA (V3.2->V4) | « mon propre sélecteur est devenu le goulot » | **profiler sa propre solution et attaquer le NOUVEAU goulot** (auto-application récursive) |
| PagedAttention | le KV cache est un problème d'allocation mémoire | **importer une abstraction déjà résolue d'un autre domaine** (mémoire virtuelle des OS) |
| EAGLE-3 | la perte de prédiction de features n'était pas nécessaire | **retirer une contrainte supposée indispensable** |
| Désagrégation P/D | prefill et decode sont deux charges hétérogènes | **désagréger selon un axe d'hétérogénéité de ressource** |
| COBS | tous les sélecteurs sont au premier ordre | **nommer l'objet mathématique approximé, puis demander à quel ORDRE en est l'état de l'art** |

[INFÉRENCE] Ce que j'ai appliqué, explicitement :
- pour le **coreset** : la méthode COBS, mais en changeant de FAMILLE d'approximation
  (support au lieu de moments) plutôt que d'ordre ;
- pour **ASP** : la méthode PagedAttention — importer une abstraction résolue ailleurs
  (identification des meilleurs bras, élimination successive) ;
- pour la **découverte des ex aequo** : la méthode DeepSeek — profiler ma propre solution,
  et constater que le goulot n'était plus l'estimateur mais la structure du problème.

Ces méthodes sont, à mes yeux, le vrai livrable transférable : elles s'appliquent à
n'importe quel sous-problème du domaine.

### Blocage externe identifié

[BLOCAGE] Le test décisif d'ASP — un noyau CUDA à deux passes, pour savoir si le surcoût de
`gather` irrégulier annule le gain en octets — demande un GPU. La machine n'en a pas. C'est
un blocage réel, pas une impression d'avoir fait le tour. Je le note et je poursuis sur les
pistes qui n'en dépendent pas.

### Deuxième hypothèse implicite partagée : les blocs sont CONTIGUS EN POSITION

[FAIT] NSA, Quest, COBS, CSA, AsyncTLS, HiSparse : tous partitionnent le KV cache en blocs
**contigus en position**. La justification est matérielle (accès mémoire contigus), jamais
sémantique.

[INFÉRENCE] Or j'ai mesuré en session 2 que la masse d'attention n'est PAS groupée
spatialement : autocorrélation 0,13 à 0,21 au décalage 1, et 30 à 47 % des tokens du top-16
sont des aiguilles isolées. Un bloc contigu est donc un agrégat **sémantiquement arbitraire**.
Il mélange des clés sans rapport, ce qui gonfle son rayon intra-bloc rho — or rho est
exactement la quantité qui contrôle l'erreur de tout résumé (théorème du POC, borne (*)).

[HYPOTHÈSE] **Si l'on définissait les blocs par le CONTENU plutôt que par la position**
(regroupement des clés similaires, une fois pour toutes au prefill), chaque bloc serait
interne­ment cohérent, rho s'effondrerait, et tous les résumés deviendraient bien plus
précis à budget égal. Le prix est la perte de contiguïté mémoire — mais un cache paginé
(vLLM) est déjà non contigu au niveau des pages, donc rassembler un bloc défini par le
contenu n'est pas structurellement plus coûteux, tant que le bloc lui-même est stocké
d'un tenant.

C'est testable immédiatement et sans GPU. Test dans `12_poc/code/content_blocks.py`.

### Résultat 3 (POSITIF, le plus important de la session) : blocs par contenu

[FAIT] À nombre de blocs et taille de bloc IDENTIQUES (L=64, top-8, blocs distants) :

| banc | rho | sigma_disc | **masse ABSOLUE captée par le top-8** |
|---|---:|---:|---:|
| Qwen sans RoPE — position | 12,72 | 0,488 | 21,61 % |
| Qwen sans RoPE — **contenu** | **8,42** (×0,66) | **0,444** (×0,91) | **43,14 %** (**×2,00**) |
| SmolLM2 sans RoPE — position | 11,86 | 0,577 | 19,14 % |
| SmolLM2 sans RoPE — **contenu** | **7,90** (×0,67) | **0,459** (×0,80) | **45,20 %** (**×2,36**) |
| Qwen avec RoPE — position | 14,20 | 0,877 | 44,55 % |
| Qwen avec RoPE — **contenu** | 13,33 (×0,94) | 0,887 (×1,01) | **54,78 %** (×1,23) |
| SmolLM2 avec RoPE — position | 13,27 | 1,097 | 51,78 % |
| SmolLM2 avec RoPE — **contenu** | 12,19 (×0,92) | **0,817** (×0,75) | **71,04 %** (×1,37) |

[DÉRIVATION] Le mécanisme prévu est confirmé : rho chute de ~1/3 et sigma_disc suit. La borne
(*) du POC prédisait exactement ce couplage.

[INFÉRENCE] Traduction opérationnelle. En blocs par position (Qwen sans RoPE), il faut
environ k=20 blocs pour capter 43 % de la masse distante (mesure adaptive_k : k=16 -> 37,4 %,
k=24 -> 49,4 %). Les blocs par contenu l'atteignent avec **k=8**. Soit **~2,5× moins de
tokens lus à qualité égale**.

### Surprise : regrouper sur le contenu PRÉ-RoPE est PIRE

[FAIT] J'attendais que le regroupement sur les clés pré-RoPE (contenu pur, indépendant de la
position) soit supérieur. Mesure sur Qwen avec RoPE :
- position : rho 14,20, masse 44,55 %
- contenu **post**-RoPE : rho 13,33, masse 54,78 %
- contenu **pré**-RoPE : rho 14,84, masse 48,79 %  <- moins bon

[INFÉRENCE] Explication. L'attention opère sur les clés POST-RoPE ; c'est donc cette
géométrie-là qu'il faut rendre compacte. Regrouper avant rotation réunit des clés
sémantiquement proches qui, tournées de façon différente selon leur position, se ré-étalent
après rotation (rho remonte à 14,84). **Il faut regrouper sur ce qui est effectivement
mis en cache : les clés post-RoPE.**

[INFÉRENCE] Corollaire plus profond : **RoPE combat activement le découpage par contenu.**
Il injecte de la position dans la géométrie des clés, ce qui explique que le gain tombe de
×2,0-2,4 (sans RoPE) à ×1,2-1,4 (avec RoPE). C'est le pendant, vu de l'autre côté, de la
découverte NoPE de COBS : eux retirent RoPE du RÉSUMÉ pour qu'il ne dépende que du contenu ;
je constate que RoPE dégrade aussi le PARTITIONNEMENT. Les architectures à RoPE découplée
(MLA, DSA : un grand latent sans RoPE + 64 dims RoPE) sont donc structurellement les mieux
placées pour en profiter.

### Limites honnêtes de ce résultat

1. [LIMITE] **Coût du regroupement.** Mon k-means équilibré glouton est en O(N·nb·log nb) par
   itération : rédhibitoire à N=1M. Il faudrait un regroupement approché (LSH, hiérarchique,
   ou en flux). Le gain suppose que ce coût s'amortit sur toute la génération.
2. [LIMITE] **Causalité.** Un bloc par contenu mélange les positions. En décodage c'est
   inoffensif (tous les blocs candidats précèdent la requête), mais pas en prefill, où chaque
   position attend un préfixe croissant. **Le découpage par contenu est donc une
   réorganisation de la phase de décodage, appliquée après le prefill.**
3. [LIMITE] **Contiguïté mémoire.** Perdue au niveau des positions, mais un cache paginé
   (vLLM) est déjà non contigu ; il suffit de stocker chaque bloc-contenu d'un tenant.
4. [LIMITE] Deux petits modèles, contexte 8k, wikitext. Non vérifié sur modèle long-contexte.
5. [CONFOND POSSIBLE] Le doublement de la masse absolue est en partie mécanique : regrouper
   les clés similaires concentre la masse sur moins de blocs. Mais c'est précisément le but
   d'un partitionnement pour l'attention creuse — et la comparaison se fait à **nombre de
   tokens lus identique**, donc elle est loyale.

---

## Session 4 — levée du blocage GPU

### Le blocage n'en était pas un : sonder avant de supposer

[FAIT] Le blocage était formulé « il faut un noyau CUDA, donc un GPU NVIDIA ». En sondant
la machine : pas de ROCm/HIP (`hipcc`, `rocminfo`, `hipconfig`, `rocm-smi` absents), mais
**`clinfo` présent et le pilote Adrenalin exposant l'iGPU en OpenCL 2.1 sous le nom exact
`gfx1152`**. OpenCL suffit à mesurer des motifs d'accès mémoire.

[INFÉRENCE] Leçon de méthode : le blocage venait d'une contrainte que je m'étais imposée
(« CUDA »), pas de la machine. Sonder l'environnement avant de planifier une installation
a économisé toute la piste ROCm.

### Argument analytique posé AVANT la mesure

[DÉRIVATION] Le gather d'ASP n'a pas une granularité de 4 octets : chaque élément gathéré
est un résumé de bloc entier, `r*D*2` octets en fp16 — 1024 o pour r=8, D=64, soit **16
lignes de cache**. La coalescence est donc parfaite À L'INTÉRIEUR du morceau ; seule la
localité entre morceaux est perdue. Prédiction : la pénalité doit s'annuler quand la
taille de morceau dépasse largement la ligne de cache.

### Mesure 1 — la prédiction est confirmée, et le seuil est mesuré

[FAIT] Ratio gather/contigu sur gfx1152, volume identique, buffer 768 Mio :
64 o → 0,67 | 256 o → 0,73 | 512 o → 0,83 | **1 Kio → 0,78-1,07** | **2 Kio → 1,01** |
16 Kio → 0,99. **Bascule entre 1 et 2 Kio.**

### Détour instructif : mon premier banc concluait l'inverse

[AUTOCORRECTION] La v1 du banc ASP donnait une efficacité médiane de **46,7 %** et
2/16 configurations plus lentes que le plat — j'allais conclure que le gather tue ASP.
Le diagnostic par phase a montré que la passe 2 (le gather) tournait déjà à 52-85 GB/s,
soit la vitesse du contigu, et que le goulot était la **passe 1** à 13-53 GB/s. Or la
passe 1 est un accès CONTIGU : je l'avais implémentée avec un work-group par bloc de
128 à 512 octets, régime où le GPU ne sature pas. **Artefact du banc, pas propriété
d'ASP.** Sans la ventilation par phase, la conclusion aurait été exactement inverse de
la vérité. À retenir : toujours ventiler le temps par phase avant de conclure.

### Mesure 2 — le test décisif, les deux schémas implémentés au mieux

[FAIT] 64 configurations, deux balayages complets, gfx1152 :
- **efficacité médiane 96,2 %** (gain réel / gain théorique en octets), étendue 78-115 %
- **0/64 configurations plus lentes** que le plat
- gather 75,7 GB/s vs contigu 79,7 GB/s -> **le gather atteint 94,9 % du contigu**
- contrôle de dérive thermique : 2e balayage à 94,0 %, écart 2,3 % -> mesure stable

**VERDICT : le gather ne coûte que ~5 %. Le gain d'ASP survit.**

### Frontière de validité — une règle de conception

[FAIT] L'efficacité s'effondre quand le résumé fin descend sous ~1 Kio par bloc :
à S2=512 o (D=64, r2=4) le gain réel tombe à **0,99×**, c'est-à-dire rien.
**Règle : ASP exige `r2 * D * 2 >= 1024 octets`.** Les configurations utiles du POC
(r2 = 8 ou 16) la satisfont toutes.

### Ce que ce test ne valide pas

[LIMITE] Le 860M est un iGPU RDNA 3.5 à 4 WGP sur LPDDR5 **partagée avec le CPU**
(~80 GB/s). Un H100/MI300 a de la HBM à ~3,3 To/s — facteur **40** — avec des pages DRAM
et une hiérarchie de caches différentes. Le test valide le MÉCANISME (granularité ≫ ligne
de cache -> coalescence préservée), pas les chiffres absolus en production. Sur HBM le
seuil de 1 Kio pourrait se déplacer, plutôt vers le haut.

[BLOCAGE RÉSIDUEL] Confirmer sur NVIDIA/HBM demande d'exécuter `asp_bench.cpp` (source
unique nvcc/hipcc, écrite sans primitive de warp pour être insensible à la largeur du
wavefront) sur un GPU cloud. Marche à suivre Colab dans `16_gpu/GUIDE_SETUP.md` —
3 minutes, gratuit, mais demande une exécution côté utilisateur.

### Piste 3 — proxy CPU

[SIGNAL FAIBLE — non transposable au GPU] Même tendance qualitative sur une hiérarchie
mémoire entièrement différente : ratio 0,58 (256 o) à 0,98 (64 Kio). Contre-point, pas
validation.

### Sur ROCm

[FAIT] La documentation AMD indique que c'est **ROCm 10.0.0** qui liste le Ryzen AI 7 350
/ Radeon 860M (gfx1152), et **sous Linux** ; il n'existe pas de build gfx1152 pour le HIP
SDK Windows (issue ROCm/TheRock #1980). La référence de l'utilisateur à « ROCm 7.13,
mai 2026 » ne correspond pas à ce que j'ai pu vérifier — signalé, à confirmer.
Checklist Linux/WSL2 vérifiable étape par étape dans `16_gpu/GUIDE_SETUP.md`.
Devenue **optionnelle** puisque OpenCL a tranché.

---

## Session 5 — trois fronts

### Front 1 — modèle analytique du seuil, posé AVANT la mesure HBM

[DÉRIVATION] Le noyau de gather affecte un work-group par morceau de C octets ; le noyau
contigu balaie de grands segments. L'asymétrie n'est pas dans le motif d'adresses mais
dans le **parallélisme mémoire exposé**. Loi de Little : pour saturer B avec une latence
lambda il faut `F* = B*lambda` octets en vol. Le gather en expose `N_res * min(C, U)`,
avec N_res le nombre de work-groups résidents et U la profondeur de pipeline intra-groupe
(~6 Kio). D'où

        C* = min( U , alpha * B*lambda / N_res )

alpha = 3,28 calibré sur le seuil de 1 Kio mesuré sur gfx1152.

[DÉRIVATION] Prédictions, **contre-intuitives** : B est 40x plus grand sur HBM, mais N_res
aussi (4224 blocs sur H100 contre 128), et les deux se compensent largement.

| GPU | F* = B·λ | N_res | **seuil C\*** |
|---|---:|---:|---:|
| gfx1152 (mesuré) | 0,04 Mo | 128 | 1024 o |
| T4 | 0,13 Mo | 1280 | **328 o** |
| A100 | 1,06 Mo | 3456 | 1009 o |
| H100 | 2,01 Mo | 4224 | **1559 o** |
| MI300X | 3,44 Mo | 4864 | **2321 o** |

[INFÉRENCE] La règle `r2·D·2 >= 1024 o` tient sur T4 et A100, mais **pas sur H100/MI300X**
où il faudrait ~2 Kio. La règle sûre pour tout le parc devient `r2·D·2 >= 2048 o`.

[FALSIFICATION] Le modèle est réfuté si le seuil mesuré sur T4 est PLUS HAUT que sur le
860M (le modèle prédit 328 o contre 1024 o). Un balayage de granularité a été ajouté au
banc C++ exprès pour ce test. Notebook prêt : `16_gpu/ASP_Colab.ipynb`.
[BLOCAGE] Exécution côté utilisateur nécessaire (pas de GPU NVIDIA ici).

### Front 2 — les +5 à +8 points de la loi rappel(sigma) : RÉSOLU

Trois hypothèses testées successivement, deux réfutées.

**[RÉFUTÉ] H1 — biais différentiel par rang.** J'ai mesuré le profil complet de l'erreur
par tranche de rang vrai (moyenne et écart-type). Un modèle génératif « profil complet »
(biais + hétéroscédasticité) prédit à 1,62 point, mais le modèle « profil sans biais »
prédit à **0,59 point** : **inclure le biais mesuré DÉGRADE**. Explication : la moyenne
conditionnelle de l'erreur au rang vrai est mécaniquement négative pour tout estimateur
bruité (régression vers la moyenne) ; l'injecter dans un modèle génératif la compte deux
fois, puisque la simulation la reproduit déjà.

**[RÉFUTÉ] H2 — sur-comptage de la variance inter-strates.** Décomposition
`Var = E[Var intra] + Var[E inter]`. Prédire avec sigma_intra au lieu de sigma_all
n'améliore que marginalement (3,78 -> 3,62 point) : la variance inter-strates est petite.

**[CONFIRMÉ] H3 — inégalité de Jensen sur la dispersion de sigma ENTRE REQUÊTES.**
La courbe de référence est construite à sigma FIXE. Or sigma varie fortement d'une
requête à l'autre (||q|| varie), et R(sigma) est **convexe** dans le domaine utile, donc

        moyenne_i R(sigma_i)  >  R( moyenne_i sigma_i )

| | avec RoPE | sans RoPE |
|---|---:|---:|
| écart absolu moyen, `R(sigma moyen)` | 3,53 pts | 4,53 pts |
| écart absolu moyen, `moyenne R(sigma_i)` | **1,52 pts** | **2,75 pts** |
| **corrélation CV(sigma) ↔ écart** | **+0,956** | **+0,934** |

[FAIT] La corrélation de +0,95 entre la dispersion CV(sigma) et l'écart à la loi est
décisive, et elle explique l'ORDRE observé : COBS et GMM ont CV = 1,2 à 2,7 ; les coresets
0,7 à 1,6 ; mean-pool 0,6 à 1,1. Ce sont exactement les méthodes qui « battaient » la loi.

**Second mécanisme, mesuré indépendamment par test de permutation.** En permutant les
erreurs réelles ENTRE BLOCS à l'intérieur de chaque requête (ce qui détruit toute
structure en rang mais conserve exactement la loi marginale) :
- (i) erreur réelle **<** (ii) erreur permutée, de −0,4 à −5,2 points -> la structure en
  rang de l'erreur est **nuisible**, pas favorable ;
- (ii) permutée ≈ (iii) gaussienne de même sigma (+0,21 / −0,01 point) -> la **forme
  marginale ne compte pas** (la kurtose n'est qu'un corrélat de CV(sigma), pas une cause).

**Décomposition finale.** L'écart observé = **+3 à +9,5 points de Jensen** (dominant)
**− 1,2 à 5,2 points de mauvais placement en rang**. Appliquer la loi par requête puis
moyenner sur-corrige donc légèrement (−2,4 à −4,4 sur COBS), l'écart résiduel étant la
pénalité de structure mesurée séparément.

**[CORRECTION DE LA LOI]** Le rappel se prédit par `moyenne_i R(sigma_i)`, jamais par
`R(moyenne sigma_i)`. C'est une correction d'échelle, pas un changement de loi.
**Quest reste l'exception** : CV(sigma) faible (0,53) mais pénalité de structure la plus
forte (−3,5 à −5,2), cohérent avec son erreur de borne supérieure, énorme et mal placée.

### Front 3 — trois résultats, dont deux qui corrigent mon cadrage

**[RÉFUTÉ] Le cadrage « contenu » était faux.** J'avais dérivé la fraction de contenu
préservée par RoPE :
        phi(L) = somme_i E_i * sinc(L*theta_i) / somme_i E_i
avec theta_i = base^(-2i/D) et E_i l'énergie mesurée du bloc de fréquence i. Prédiction :
phi gouvernerait le gain du découpage par contenu. **Mesure : phi = 0,734 (Qwen) contre
0,121 (SmolLM2), alors que SmolLM2 gagne DAVANTAGE.** phi ne prédit pas le gain.

**[REFORMULATION]** L'objectif n'a jamais été la cohérence sémantique mais la **compacité
géométrique dans l'espace post-RoPE**, celui où l'attention opère et où rho se mesure.
« Blocs par le contenu » est un abus de langage : il faut dire **blocs définis par la
géométrie des clés effectivement mises en cache**. Cela explique directement pourquoi le
regroupement pré-RoPE est moins bon (×1,08-1,10) : il optimise dans le mauvais espace.

**[CONFIRMÉ] phi prédit autre chose — la nature des groupes.** Étalement positionnel des
groupes, rapporté à un tirage uniforme (0 = contigu, 1 = aléatoire) :
- Qwen (phi = 0,73) : étalement **0,405** -> groupes mi-contenu mi-position
- SmolLM2 (phi = 0,12) : étalement **0,076** -> groupes quasi POSITIONNELLEMENT LOCAUX

Quand RoPE domine la géométrie, le regroupement post-RoPE retrouve de lui-même des
groupes locaux. phi mesure donc la part contenu/position du regroupement, pas son gain.

**[RÉFUTÉ] Le regroupement sur le sous-espace lent échoue.** Ne garder que les blocs de
fréquence à rotation lente (L*theta_i < tau) donne des groupes positionnellement
aléatoires (étalement 0,72-0,93) et **0,33× à 0,61×** de la masse captée. Il faut la
géométrie post-RoPE COMPLÈTE. Le schéma « ni RoPE ni NoPE » que je cherchais n'existe
pas sous cette forme.

**[FAIT] Balayage des paramètres — le gain est robuste en signe, pas en amplitude.**

| | L=32 | L=64 | L=128 | | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Qwen | 1,12× | 1,08× | 1,03× | | 1,14× | 1,10× | 1,08× | 1,07× | 1,03× |
| SmolLM2 | 1,37× | 1,29× | 1,21× | | 1,70× | 1,45× | 1,29× | 1,16× | 1,07× |

Le gain **décroît avec la taille de bloc et surtout avec k** : à k=32 il ne reste que
1,03-1,07×. Le découpage par géométrie sert donc aux **petits budgets**, là où le choix
des blocs est critique — pas aux gros, où l'on capte la masse de toute façon.

**[ARTEFACT DIAGNOSTIQUÉ] Ventilation d'un écart inexpliqué.** Le gain Qwen était tombé
de 1,21× à 1,08× entre deux runs. Deux facteurs avaient changé ensemble : l'algorithme
de k-means (exact glouton -> approché vectorisé) et l'échantillonnage des couches. En
croisant les deux facteurs, tout le reste fixe :

| couches | k-means | gain Qwen | gain SmolLM2 |
|---|---|---:|---:|
| toutes | exact | **1,21×** | 1,26× |
| toutes | rapide | 1,20× | 1,26× |
| 1 sur 2 | exact | 1,11× | 1,29× |
| 1 sur 2 | rapide | 1,11× | 1,28× |

**L'algorithme n'y est pour rien** (écart ≤ 0,01×) ; c'est le **choix des couches** qui
fait tout l'écart sur Qwen (1,21 -> 1,11). Nouveau fait au passage : **le gain dépend
fortement de la couche**, ce qui n'avait pas été mesuré. Même réflexe que pour l'artefact
de la passe 1 en session 4 : ventiler par facteur avant de conclure.

---

## Session 6 — mesure sur Tesla T4 : la prédiction est partiellement RÉFUTÉE

### Mise en place

[FAIT] `google-colab-cli` installé dans WSL2, authentifié. Deux bugs rencontrés et corrigés,
notés parce qu'ils ont valeur informative :
1. **OAuth** — `oauthlib` levait une exception sur un changement de scope (`drive.file`
   absent de la réponse Google). Corrigé par `OAUTHLIB_RELAX_TOKEN_SCOPE=1`.
2. **Dépendance non épinglée** — colab-cli 0.6.0 déclare `jupyter-kernel-client` sans
   contrainte ; la 1.0.2 a renommé `KernelClient` en `JupyterKernelClient`. Patché par un
   alias rétrocompatible (`getattr(..., 'KernelClient', None) or JupyterKernelClient`),
   sauvegarde dans `runtime.py.orig`.
3. **Segfault du banc** — mon balayage de granularité lance jusqu'à 4,2 M de work-groups à
   64 o, alors que le tampon `sink` n'avait que 2^19 cases : débordement. Corrigé par un
   masque d'index. Ce bug n'existait pas sur l'iGPU car le balayage y avait été fait en
   OpenCL, sans cette partie. **Un banc porté doit être re-testé, pas supposé équivalent.**

### Résultat 1 — le gather ne coûte rien EN RÉGIME SATURÉ

[FAIT] Tesla T4 (40 SM, bus 256 bits, pic mesuré 274 GB/s) :

| morceau | contigu GB/s | gather GB/s | ratio | saturé ? |
|---:|---:|---:|---:|---|
| 64 o | 14,5 | 14,5 | 1,002 | non — borné par le lancement |
| 128 o | 42,4 | 40,0 | 0,943 | non |
| 256 o | 83,1 | 75,5 | 0,908 | partiel |
| 512 o | 167,7 | 147,6 | 0,880 | partiel |
| **1024 o** | 235,3 | 232,5 | **0,988** | **oui** |
| 2048 o | 274,0 | 260,0 | 0,949 | oui |
| 4096 o+ | ~270 | ~269 | 0,99-1,00 | oui |

**En régime saturé, ratio médian gather/contigu = 0,991.** Le gather coûte ~1 % sur T4,
contre ~5 % sur gfx1152. Le résultat central tient sur une seconde architecture.

### Résultat 2 — ma prédiction quantitative est FAUSSE

[RÉFUTÉ] J'avais prédit, avant la mesure, `C* ≈ 328 o` sur T4 par le modèle MLP
(loi de Little, `C* = min(U, alpha·B·lambda/N_res)`), soit **plus bas** que les 1024 o de
l'iGPU. **Mesure : le seuil de saturation est à 1024 o sur T4 aussi.** Le modèle se trompe
d'un facteur ~3 sur la magnitude.

Le critère de falsification strict que j'avais posé (« pénalité PLUS FORTE que sur le 860M
à 512 o ») n'est pas atteint : à 512 o, T4 donne 0,880 contre 0,83 sur l'iGPU, donc
légèrement mieux. La **direction** du modèle est faiblement confirmée, sa **calibration**
ne l'est pas.

[INFÉRENCE] Le seuil vaut ~1 Kio sur **deux systèmes mémoire très différents** (LPDDR5
partagée à 80 GB/s, 8 CU ; GDDR6 à 274 GB/s, 40 SM). Une invariance pareille est ce que
prédit un argument de **granularité DRAM** (pages/bursts de l'ordre du Kio sur les deux),
pas la loi de Little qui prédisait un facteur 3 d'écart. **J'avais mentionné cet effet
DRAM puis l'avais écarté en le supposant absorbé par le parallélisme de bancs : c'était
l'erreur.** L'explication DRAM est mieux soutenue par les données.

Conséquence pratique **favorable** : la règle `r2·D·2 >= 1024 o` n'a PAS besoin d'être
relevée à 2 Kio sur HBM comme le modèle MLP le suggérait. Mais HBM reste non testé.

### Résultat 3 — un second facteur, invisible sur l'iGPU

[FAIT] Efficacité ASP sur T4 : **médiane 79,2 %** (contre 96,2 % sur gfx1152), 2/32
configurations plus lentes. Ventilation par facteur :

| taille du résumé fin S2 | efficacité médiane | plus lentes |
|---:|---:|---:|
| 1024 o | **50,0 %** | 2/4 |
| 2048 o | 77,2 % | 0/12 |
| 4096 o | 85,2 % | 0/12 |
| 8192 o | **90,3 %** | 0/4 |

| volume total lu | efficacité médiane |
|---|---:|
| < 8 Mo | **28,3 %** |
| 8–64 Mo | 78,4 % |
| > 64 Mo | **85,2 %** |

[INFÉRENCE] **Deux contraintes, pas une.** La granularité (S2) était connue ; le
**volume total** est nouveau et n'apparaissait pas sur l'iGPU, qui sature avec bien moins
de travail. Sur un gros GPU, la structure à deux noyaux d'ASP paie un double coût de
lancement, qu'il faut amortir. Règles révisées :

    (1) resume fin :   r2 * D * 2  >=  2048 octets     (1024 o ne donne que 50 %)
    (2) volume lu   :   n * S2      >=  8 Mo            (sinon borne par le lancement)

[INFÉRENCE] **Le régime de déploiement réel satisfait largement les deux.** À 1M de
contexte avec L=64 : n = 15 625 blocs ; avec r2=16 et D=128, S2 = 4096 o et le volume vaut
64 Mo. Les échecs se concentrent sur les petits n, c'est-à-dire les contextes courts — là
où ASP n'a de toute façon aucun intérêt. Seuil d'utilité estimé : **~130 k tokens de
contexte** sur un GPU de classe T4.

---

## Session 7 — A100 PCIe 40 Go (HBM2e) : mes DEUX modèles sont réfutés

Instance vast.ai (A100-PCIE-40GB, CUDA 12.8, sm_80, 108 SM, bus 5120 bits, pic mesuré
**1372 GB/s**). Accès par clé SSH — la clé API fournie n'avait pas les privilèges 2FA,
donc ni `vastai execute` ni `create ssh-key` ; clé ajoutée manuellement côté utilisateur.

### Résultat 1 — aucune pénalité de gather, à AUCUNE granularité

[FAIT] Balayage sur A100 :

| morceau | contigu | gather | ratio | % du pic | régime |
|---:|---:|---:|---:|---:|---|
| 64 o | 55,3 | 55,3 | 1,000 | 4 % | borné par le lancement |
| 512 o | 438,4 | 438,4 | 1,000 | 32 % | partiel |
| 1024 o | 862,3 | 848,4 | 0,984 | 63 % | partiel |
| 2048 o | 1272,5 | 1236,5 | 0,972 | 93 % | saturé |
| 4096 o+ | ~1370 | ~1340 | 0,955–1,031 | ~100 % | saturé |

**Ratio minimum sur tout le balayage : 0,955.** Aucune pénalité mesurable.

[PRÉCISION] En dessous de 512 o le débit ne dépasse pas 16 % du pic : ni le contigu ni le
gather ne saturent, le ratio de 1,000 y est donc **non informatif**. La plage exploitable
commence à 512 o. À ce point de charge relative comparable (32 % du pic), le T4 donnait
0,880 et l'A100 donne 1,000 : la comparaison est loyale et l'A100 est bien meilleur.

### Résultat 2 — les deux modèles tombent

[RÉFUTÉ] **Modèle MLP** (loi de Little) : prédisait `C* = 1009 o` sur A100. Observé :
aucun seuil au-dessus de 512 o. Faux quantitativement — pour la seconde fois (328 o
prédits sur T4, 1024 o observés).

[RÉFUTÉ] **Hypothèse d'invariance DRAM**, que j'avais adoptée après le T4 : le seuil
serait ~1 Kio partout. Observé : il **disparaît** sur A100. Il n'est donc pas invariant.

[INFÉRENCE] Ce qui survit : la **direction** du modèle MLP. La pénalité décroît
monotonement avec le parallélisme du GPU —
gfx1152 (8 CU) : 0,67 à 64 o · T4 (40 SM) : 0,88 à 512 o · A100 (108 SM) : 1,00 à 512 o.
Plus le GPU est parallèle, plus le gather est bénin. Seule la calibration était fausse.
**Conclusion pratique favorable : sur GPU datacenter, le gather d'ASP est gratuit.**

### Résultat 3 — le vrai obstacle est le coût de lancement, et c'est un artefact de mon banc

[FAIT] Efficacité ASP brute sur A100 : **médiane 30,6 %, 16/32 configurations plus lentes**
que le plat. Bien pire que T4 (79,2 %) et gfx1152 (96,2 %).

[DÉRIVATION] Diagnostic par ventilation : `t_p1` et `t_p2` ne descendent jamais sous
**11,3 µs**, quelle que soit la configuration — c'est un plancher, donc le coût de
lancement d'un noyau. Or sur A100 le schéma plat ne prend que 12 à 30 µs dans la plupart
des configurations : le second lancement d'ASP suffit à tout annuler.

Par volume total lu :

| volume | efficacité brute |
|---|---:|
| < 16 Mo | 22,2 % |
| 16–64 Mo | 30,6 % |
| **> 64 Mo** | **77,6 %** |

Meilleures configurations : 134 Mo -> **94,2 %** d'efficacité (gain réel 2,51×) ;
67 Mo -> 87,5 %.

[ABANDON] J'ai tenté de corriger analytiquement en retranchant le plancher de 11,3 µs des
deux schémas. **Correction inutilisable** : pour les petites configurations le temps total
EST l'overhead, la soustraction divise par un résidu quasi nul et produit des efficacités
absurdes (9413 %). Je m'en tiens aux chiffres bruts et je signale l'artefact plutôt que de
le maquiller.

[LIMITE MÉTHODOLOGIQUE, importante] **Mon banc mesure ASP comme DEUX LANCEMENTS DE NOYAU
SÉPARÉS.** Une implémentation réelle fusionnerait les deux passes dans un seul noyau (ou
utiliserait un graphe CUDA / un noyau persistant), supprimant le second lancement. Le coût
de lancement est donc un artefact de la conception du banc, pas une propriété d'ASP — mais
je n'ai pas mesuré de version fusionnée, donc **je ne peux pas affirmer que le gain est
récupéré**. C'est le prochain test à faire.

### Bilan des trois GPU

| | gfx1152 (LPDDR5) | T4 (GDDR6) | A100 (HBM2e) |
|---|---:|---:|---:|
| pic mesuré | 80 GB/s | 274 GB/s | 1372 GB/s |
| pénalité de gather | ~5 % | ~1 % | **0 %** |
| seuil de granularité | ~1–2 Kio | ~1 Kio | **aucun ≥512 o** |
| efficacité ASP médiane | 96,2 % | 79,2 % | 30,6 % (brute) |
| volume requis | quelques Mo | ~8 Mo | **~64–134 Mo** |

[INFÉRENCE] Lecture d'ensemble : **la question posée initialement est tranchée par la
négative — le gather ne coûte rien, et de moins en moins à mesure que le GPU grossit.**
Mais un second obstacle, invisible sur petit matériel, le remplace : plus le GPU est
rapide, plus la structure à deux passes doit brasser de données pour amortir ses
lancements. Ce n'est pas fatal (le régime 1M de contexte donne 64 Mo, soit 78-94 %
d'efficacité observée), mais cela déplace le travail d'ingénierie du *motif d'accès* vers
la *fusion des noyaux*.

---

## Session 8 — fusion du noyau : conception avant mesure

### Le coût à supprimer, chiffré

[FAIT] Session 7 : `t_p1` et `t_p2` ne descendent jamais sous **11,3 µs** sur A100 — le
coût de lancement d'un noyau. Le schéma à deux lancements paie donc `2L` là où le plat
paie `L`. Quand `t_flat` vaut 12–30 µs, ce `L` supplémentaire annule tout.

Modèle : `T_2noyaux = 2L + t_p1 + t_p2` contre `T_fusionne = L + t_p1 + t_p2 + S`,
avec `S` le coût de synchronisation interne. La fusion gagne ssi **S < L**.

### Le problème structurel de la fusion, et trois conceptions

[DÉRIVATION] La passe 2 a besoin du top-m produit par la passe 1 : c'est une **dépendance
globale** entre tous les blocs d'une même requête. Dans un noyau unique, trois façons de
la satisfaire :

**(A) Un work-group par requête.** Chaque groupe traite les n blocs de sa requête, garde
son top-m en mémoire partagée, puis gathère. Zéro synchronisation. **Mais le
parallélisme tombe au nombre de requêtes** `Q = batch × têtes KV`. En décodage avec
batch 32 et GQA 8 têtes, Q = 256 groupes pour 3456 emplacements résidents sur A100 :
**7 % d'occupation**. De plus chaque groupe doit streamer seul `n·S1` = 8 Mo. Rédhibitoire.

**(B) Synchronisation de grille** (`cooperative_groups::grid.sync()`). Préserve le
parallélisme mais coûte 5–10 µs sur A100 — **du même ordre que le lancement qu'on
cherchait à éviter**. Le gain s'évapore.

**(C) Sélection STRATIFIÉE — la conception retenue.** Observation : ASP est un *filtre
heuristique*, pas une sélection exacte. On n'a donc pas besoin du top-m global. On
découpe les n blocs en G strates, chaque work-group prend le **top-(m/G) de sa strate** et
gathère immédiatement ses propres survivants. **Zéro synchronisation, parallélisme complet,
un seul lancement.**

[COMPROMIS À QUANTIFIER] La sélection n'est plus le top-m global mais une union de tops
locaux. Comme les blocs sont ordonnés arbitrairement vis-à-vis de leur pertinence, les
strates sont échangeables et l'approximation devrait être faible — mais **c'est une
hypothèse, pas une évidence** : si la masse d'attention se concentre dans une strate, le
quota `m/G` la tronque. C'est mesurable sur le banc de rappel existant, sans GPU.
Je le quantifie AVANT d'implémenter le noyau.

### Prédiction posée avant mesure

[HYPOTHÈSE] (1) La fusion supprime exactement un `L`, soit ~11,3 µs sur A100 : les
configurations à faible volume doivent passer d'efficacité ~22 % à ~45-60 %.
(2) Le coût de la stratification en rappel doit croître avec G et décroître avec m.
(3) Un résidu subsiste probablement : la frontière de code plus grosse (deux phases dans
un noyau) augmente la pression registre et peut réduire l'occupation. **Si l'efficacité
fusionnée ne remonte pas comme prévu, c'est la première piste à ventiler.**


---

## Fin de session — consolidation v2

- Re-vérification à la source de tous les chiffres v1 ; corrections : recette 943,7B → V3.2 ; « 70 Ko/token » → [INFÉRENCE] (68,6 Kio calculés) ; « PD-Pai » → TaiChi ; plage EAGLE-3 « 1,9×–3,6× » → chiffres réels (5,51× / 158,34→373,25 / 1,38× b=64 / 0,88× / 1,01×).
- Addenda v2 dans axe1–axe4 ; série T fusionnée dans QUESTIONS_OUVERTES (10 Q + 10 T).
- Corpus : 31 → 38 PDFs en session ; Mistletoe extrait et vérifié (5,47×→1,83× GSM8K ; MT-bench ÷1,89).
- v1 archivée dans `_archive_v1/` ; copies finales à la racine.

### Fusion, première tentative : DEUX artefacts de banc, diagnostiqués par ventilation

[FAIT] Premier résultat : la fusion est **20 à 50× plus lente** que les deux noyaux
(médiane 0,02×, 24/24 configurations perdantes). Résultat absurde -> ventilation avant
toute conclusion.

**[ARTEFACT 1 — comparaison invalide, le plus grave]** Le ratio observé (0,02–0,05)
vaut à peu près `1/Q` avec Q=64. Ce n'est pas un hasard : dans mon banc, les chemins
*plat* et *deux noyaux* lisent les résumés **une seule fois au total**, alors que le
noyau fusionné les lit **une fois PAR REQUÊTE** (Q=64 groupes de requêtes). Je comparais
33 Mo contre 537 Mo.

C'est une erreur de modélisation de ma part, et elle révèle quelque chose d'important sur
le problème lui-même : **dans un vrai pas de décodage, chaque requête (batch × têtes KV)
doit scanner TOUS les n blocs.** Le travail du sélecteur est donc `Q · n · S1`, pas
`n · S1`. Mon banc à deux noyaux des sessions 4 à 7 mesurait implicitement Q=1 — les
gains y restent valides comme *rapports* entre schémas à Q fixé, mais les volumes absolus
étaient sous-estimés d'un facteur Q.

**[ARTEFACT 2 — non-coalescence en phase 1]** Dans mon noyau fusionné, chaque THREAD lit
un résumé de bloc entier en série (`for i<c1`), et deux threads voisins lisent des
adresses distantes de `G·c1` uint4, soit 4 Kio. Accès totalement non coalescé. C'est
**exactement la même classe d'erreur que l'artefact de la passe 1 en session 4** (un
work-group par petit morceau alors), transposée d'un cran plus bas (un thread par bloc).
Correction : faire coopérer une **équipe de TW=32 threads** par résumé de bloc, ce qui
rétablit la coalescence, au prix d'une barrière par itération.

[INFÉRENCE] Ces deux artefacts se cumulaient. Aucune conclusion sur la fusion n'est
possible avant de les corriger : je reprends le banc avec (1) les trois chemins faisant
le même nombre Q de requêtes, (2) une phase 1 coalescée par équipes.

### Fusion v2 : deux artefacts corriges, et la decouverte que l'iGPU ne peut pas trancher

[FAIT] Apres correction des deux artefacts (charge egale a Q requetes ; phase 1 coalescee
par equipes de 32 threads), la fusion passe de 0,02x a **0,49x** : elle reste perdante
sur les 24 configurations.

[FAIT] Troisieme optimisation testee : insertion a SEUIL du top-q (le cas courant devient
O(1) au lieu de O(q), ce qui supprime un goulot serie sur une seule voie pendant que 255
threads attendent la barriere). Gain : **0,53x -> 0,49x, soit rien.** Ce n'etait donc pas
le goulot non plus. [ABANDON de cette piste d'optimisation.]

### Le point decisif : l'iGPU est structurellement incapable de tester la fusion

[DERIVATION] La fusion ne sert qu'a supprimer UN cout de lancement `L`. Son gain relatif
maximal vaut donc

        gain_max = L / T_noyau

  - Sur A100 : L = 11,3 us et les noyaux durent 12 a 30 us -> gain_max = **38 a 94 %**.
    C'est la que la fusion se joue, et c'est bien la que le probleme etait apparu.
  - Sur gfx1152 : L ~ 10 us mais les noyaux durent 0,5 a 12 **ms** -> gain_max = **0,1 a
    2 %**, noye dans le bruit. En revanche la fusion y paie tous ses couts (barrieres
    supplementaires, serialisation de la selection, occupation reduite).

**Sur cet iGPU, la fusion ne peut donc que perdre, par construction.** Mesurer 0,49x n'y
refute rien : c'est la mesure d'un cout sans le benefice correspondant. J'aurais du poser
ce calcul avant de lancer le banc — c'est le meme reflexe (argument analytique d'abord)
que j'avais correctement applique pour la granularite du gather, et que je n'ai pas
applique ici.

[LIMITE DE BANC, a corriger sur GPU rapide] Un troisieme probleme de modelisation subsiste :
pour les chemins plat et 2-noyaux je multiplie par Q une mesure a une requete, ce qui
multiplie aussi leur cout de lancement par Q. Un vrai systeme lancerait un seul noyau a
grille Q fois plus grande. Il faut donc, sur le GPU cible, mesurer les trois chemins avec
une grille reellement dimensionnee a Q, sans multiplication a posteriori.

[CONCLUSION D'ETAPE] Le volet 1 ne peut pas etre conclu sans un GPU rapide. Demande
materielle formulee a l'utilisateur.

---

## Session 9 — H200 : la fusion FONCTIONNE, à condition de choisir q

Instance vast.ai **NVIDIA H200 NVL** (132 SM, bus 6016 bits, 143 Go, CUDA 12.8, sm_90).
Banc CUDA `16_gpu/fused.cu`, trois chemins à grilles réellement dimensionnées à Q.

### Trois bugs trouvés par bissection, tous consignés

**[BUG 1 — `cudaErrorInvalidAddressSpace`]** Le noyau fusionné échouait dès le premier
lancement. Bissection en quatre variantes (shfl seul / + top-q partagé / + fusion finale /
+ phase 2) : seule la phase 2 échouait. Cause : **j'initialisais `bs`/`bi` uniquement pour
`t < q`**, alors que la fusion finale balaie les `NT*QMAX = 512` entrées. Les 256 cases non
initialisées contenaient des restes des noyaux précédents -> indices de bloc aberrants ->
accès hors limites. Correction : initialiser tout le tableau.

**[BUG 2 — goulot sériel]** Après correction, le temps fusionné était **exactement linéaire
en q** (0,13 ms à q=16 ; 0,25 ms à q=32) et **indépendant de la taille des résumés** —
signature d'une sérialisation, pas d'un problème mémoire. Cause : la fusion finale des
tops locaux fait `q × NT × QMAX` itérations **sur un seul thread**. Correction : supprimer
la fusion finale en stratifiant d'un cran de plus — chaque équipe gathère directement son
propre top-(q/NT), sans merge ni synchronisation. Gain immédiat : 0,31x -> 0,96x.

**[BUG 3 — dépassement de tampon du banc]** À Q=64 et S2=8 Kio, le chemin plat lit
`Q·n·S2` = 8,6 Go dans un tampon de 4 Go. Les 59 configurations mesurées avant ce point
restent valides ; à corriger par un plafond sur les configurations.

### Résultat : la fusion gagne, mais seulement à faible q

[FAIT] 59 configurations, Q=64 requêtes, G balayé dans {32, 128, 512} :

| q | configurations | fusion / 2-noyaux (médiane) | gain vs plat | gagnantes |
|---:|---:|---:|---:|---:|
| **8** | 15 | **1,57×** | **1,71×** | **15/15** |
| 16 | 15 | 0,78× | 1,27× | 3/15 |
| 32 | 15 | 0,68× | 0,71× | 0/15 |
| 64 | 14 | 0,32× | 0,38× | 0/14 |

**À q=8 la fusion bat le schéma à deux noyaux dans 15 cas sur 15, de 1,57× en médiane**,
et le gain total contre le schéma plat atteint **4,58× au maximum** (médiane 1,71×, contre
1,17× pour le 2-noyaux seul sur les mêmes configurations).

[DÉRIVATION] Pourquoi q=8. Le produit `G·q = m` est fixé : G gouverne le parallélisme
(nombre de work-groups), q le travail sériel par équipe (insertion du top-q, puis gather de
q/NT blocs). Diminuer q augmente donc le parallélisme à volume constant. L'optimum est
atteint quand le quota par équipe `qt = q/NT` tombe à 1, soit **q ≈ NT = 8**, d'où la règle
de conception **G ≈ m/8**.

### Le compromis, quantifié

[FAIT] À q=8 avec NT=8 équipes, le quota par strate vaut exactement 1 — la stratification
la plus fine. La mesure hors GPU (`12_poc/code/stratified.py`) donne pour ce régime une
perte de **2,5 à 3,2 points relatifs** de rappel de masse en mode entrelacé. C'est le prix
de la fusion, et il est connu avant déploiement, pas découvert après.

[INFÉRENCE] Bilan du volet 1 : **l'artefact du double lancement est bien résolu** — non
par la fusion naïve (qui perd), mais par une fusion dont le paramètre de stratification est
choisi pour maximiser le parallélisme. La prédiction de la session 8 (« la fusion supprime
un L, donc +38 à 94 % sur A100 ») était qualitativement juste mais pour une mauvaise
raison : le gain mesuré (1,57×) vient surtout de l'augmentation du parallélisme permise par
la stratification fine, pas seulement du lancement économisé.

---

## Session 10 — Volet 2 : ASP dans Qwen3-8B, résultat NÉGATIF et sa cause

Instance H200 NVL, Qwen3-8B (36 couches, 32 têtes Q, 8 KV, head_dim 128), poids réels
15,3 Go, intégration par `ALL_ATTENTION_FUNCTIONS["asp"]` — le point d'extension prévu
par transformers 5.16, pas un monkeypatch.

### Contraintes matérielles qui ont dicté le choix du modèle

[FAIT] Le disque de l'instance ne fait que **32 Go**, ce qui exclut tout modèle au-delà de
~20 Go de poids. Par ailleurs **tous les Qwen3 ont `max_position_embeddings = 40960`** :
au-delà de 40k on sort du contexte d'entraînement (il faudrait YaRN). On mesure donc de la
**performance**, pas de la qualité, au-delà de 40k — le motif mémoire et la latence sont
rigoureusement identiques, seules les sorties perdent leur sens.

### Un artefact de mesure qui aurait inversé la conclusion

[AUTOCORRECTION] Ma première ventilation donnait une attention dense de **121 ms** à 131k
contre 21 ms pour ASP, soit un gain de 5,8× — et à 262k un gain de 10,3×. Résultat
impossible : l'attention seule (121 ms) dépassait le pas complet mesuré (30 ms).

Cause : dans les deux chemins je matérialisais le KV étendu par `repeat_interleave(g)`
pour émuler GQA, ce qui **gonfle la baseline dense d'un facteur g=4** en mémoire lue. Le
chemin ASP, lui, ne gathère qu'un KV minuscule, donc l'inflation ne le touchait quasiment
pas. **L'artefact favorisait massivement ASP.** Correction : `enable_gqa=True` dans SDPA
des deux côtés.

Après correction, l'attention dense tombe à 2,06 / 4,06 / 6,62 / 11,34 ms (16k → 262k),
soit **~84 % de la bande passante crête** : elle est quasi optimale. C'est ASP qui coûte
**20 à 23 ms constants**.

### Résultat de bout en bout, corrigé

[FAIT] Pas de décodage complet, Qwen3-8B, H200 :

| contexte | KV | dense | ASP | gain | octets lus dense | ASP | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 k | 2,2 Go | 29,9 ms | 49,6 ms | **0,60×** | 2 416 Mo | 179 Mo | 13,5× |
| 65 k | 9,0 Go | 29,8 ms | 49,5 ms | 0,60× | 9 664 Mo | 462 Mo | 20,9× |
| 131 k | 18,0 Go | 30,0 ms | 49,4 ms | 0,61× | 19 327 Mo | 840 Mo | 23,0× |
| 262 k | 36,0 Go | 42,1 ms | 54,0 ms | 0,78× | 38 655 Mo | 1 595 Mo | 24,2× |
| 524 k | 72,0 Go | 72,6 ms | 89,8 ms | **0,81×** | 77 309 Mo | 3 105 Mo | 24,9× |

**ASP lit 13 à 25× moins d'octets et reste 1,2 à 1,7× plus LENT.**

### La cause, ventilée

[DÉRIVATION] Décomposition du pas de décodage à 131k :
- partie hors attention (MLP, normes, projections) : ~23 ms
- attention dense : 6,6 ms
- surcoût de sélection ASP : **~20 ms constants**

Le surcoût de sélection (~0,55 ms par couche × 36) vient d'une chaîne de **~8 opérations
PyTorch non fusionnées** par couche — einsum, logsumexp, topk, gather — soit ~290
lancements de noyaux par pas. C'est exactement ce que le volet 1 avait identifié comme le
facteur limitant, ici confirmé au niveau du modèle.

[DÉRIVATION] Point de croisement : ASP gagne quand `attention dense > surcoût de
sélection`. L'attention dense croît d'environ 0,043 ms par millier de tokens ; atteindre
20 ms demande **N ≈ 470 k**, et à 524 k on est encore à 0,81×. **Le croisement, dans cette
implémentation, est au-delà du million de tokens.**

### Écart à la prédiction, et ce qu'il faudrait

[FAIT] Prédiction antérieure : 10 à 40 % de gain. **Mesure : 19 à 40 % de perte.** L'écart
s'explique entièrement par une hypothèse fausse de la prédiction — elle supposait un
décodage borné par la mémoire. Il ne l'est pas ici :
- plancher mémoire théorique du pas complet (poids + KV) à 131k : 9,4 ms
- mesuré : 30 ms, soit **3,2× au-dessus du plancher**

La partie hors attention (~23 ms) est dominée par le surcoût de framework de
HF/transformers (36 couches × nombreux petits noyaux, sans graphe CUDA). Dans un moteur
optimisé (vLLM, SGLang, graphes CUDA), cette partie tomberait vers ~5 ms et l'attention
deviendrait le terme dominant — c'est le régime où les 13-25× d'octets économisés par ASP
comptent. [INFÉRENCE, non mesurée.]

### Conclusion du volet 2

[FAIT] **Dans une intégration PyTorch non fusionnée, ASP perd à toutes les longueurs de
contexte testées, malgré 13-25× d'octets économisés.** Le surcoût de sélection non fusionné
(~20 ms) dépasse à lui seul l'intégralité du temps d'attention dense (2-11 ms).

[INFÉRENCE] Ce résultat négatif **valide la direction du volet 1** : la fusion du noyau
n'est pas une optimisation, c'est une **condition nécessaire**. Le noyau fusionné mesuré
sur H200 (1,57× contre le schéma à deux noyaux) est la brique qui manque ici. Deux
conditions restent à réunir pour qu'ASP soit utile en production :
  (1) sélection en **un seul noyau fusionné** (fait, volet 1, hors modèle) ;
  (2) intégration dans un moteur où le décodage est réellement borné par la mémoire
      (non fait — HF/transformers est à 3,2× de son plancher).

---

## Session 11 — Volet 3 : rédaction du papier, et audit de traçabilité

### Choix de forme (et pourquoi)

[DÉCISION] **Lieu visé : MLSys.** Le résultat central n'est pas un gain de qualité de
modèle mais une **mesure système** : où passent les octets et les millisecondes, et
pourquoi les premiers ne se convertissent pas dans les secondes. Un tel papier serait mal
évalué à NeurIPS/ICML (pas de gain de perplexité, pas de nouveau modèle) et bien évalué
dans une communauté systèmes, qui accepte qu'un résultat négatif quantifié soit une
contribution. Format `acmart/sigconf`, présent localement et compilable sans dépendance
externe ; les fichiers de style NeurIPS/ICML ne sont pas installés et n'auraient de toute
façon pas correspondu au propos.

[DÉCISION] **Langue : anglais.** Non par convention aveugle : le corpus de référence, les
lieux visés et les relecteurs potentiels le sont. Le journal, lui, reste en français.

[DÉCISION] **Titre : « Bytes Are Not Milliseconds ».** Le résultat le plus solide et le
plus transférable de tout le travail est précisément cette dissociation. Un titre qui
annoncerait un gain serait un contresens sur le contenu.

### Audit de traçabilité — quatre nombres faux retrouvés

[AUTOCORRECTION] Avant de figer le papier, j'ai vérifié **chaque nombre** contre les
fichiers de résultats plutôt que contre ma mémoire de session. Quatre étaient faux :

1. **« σ ≤ 0,20 pour 95 % de rappel, σ ≤ 0,05 pour 99 % »** → recalculé par interpolation
   de la courbe de référence (`law_selection.json`, jeu « avec RoPE ») : **σ ≤ 0,39** et
   **σ ≤ 0,16**. Les valeurs mémorisées venaient du jeu *sans* RoPE.
2. **« incertitude résiduelle 2,589 nats, ratio 36,8 »** → fausse précision. La mesure
   (`diagnostics_regime.json`, champ `reste`) donne **1,08 nat sans RoPE, 2,36 avec**,
   soit un ratio de 15 à 34 face à l'écart d'ex æquo de 0,070 nat. La conclusion tient,
   le chiffre non.
3. **« gather : 95,1 % / 99,1 % / 98,0 % ; efficacité 50 % à 1 KiB sur T4 »** → aucune de
   ces valeurs ne se retrouve dans les données tracées. Recalcul sur le **régime saturé**
   (chemin contigu ≥ 80 % du crête) : **104,0 % / 98,4 % / 97,7 %**, jamais sous 94,9 %.
   La phrase sur l'« efficacité à 1 KiB » mélangeait deux expériences distinctes (ratio
   gather/contigu par granularité vs efficacité ASP par volume lu) : supprimée.
4. **« blocs par contenu : 1,70× à k=2, 1,07× à k=32 »** → l'expérience
   (`content_blocks.json`) est menée à **k=8 fixe**, balayée en r, pas en k. Ce balayage
   n'existe pas. Remplacé par ce qui est mesuré et qui dit mieux la même chose : le gain
   est de **×1,99 sans RoPE mais seulement ×1,23 avec**, et le regroupement pré-RoPE n'en
   récupère que **×1,10**.

[FAIT] Les autres nombres sont confirmés **exactement** : rang effectif 17,79/64 ;
r = +0,956 pour la corrélation de Jensen ; amortissement spéculatif 41,8 % (γ=8, m=1) et
65,2 % (m=4) ; accélération spéculative ramenée à 1,00× à α=0,7 contre 1,78× en dense ;
oracle − coreset = 5,42 points d'erreur de sortie contre 8,50 points pour un doublement
de k ; 21,6 % → 43,1 % pour les blocs par contenu sans RoPE ; borne coreset respectée
sur 100 % des cas.

[LEÇON] Le taux d'erreur des nombres retenus de mémoire est ici de **4 sur ~20**, tous
dans le sens qui rendait le récit plus net. C'est un biais systématique, pas du bruit :
un résumé de session conserve les conclusions et lisse les valeurs. Règle à appliquer
désormais : **aucun nombre ne va dans un livrable sans être relu dans le fichier qui le
produit**, même quand on croit s'en souvenir.

### Opinion éditoriale demandée : les blocs par contenu

[DÉCISION] **Ils ne doivent pas être dans ce papier.** Trois raisons, par ordre de poids :

1. **Ce n'est pas la même question.** ASP répond à « comment allouer la précision entre
   les blocs » ; les blocs par contenu répondent à « comment définir les blocs ». Les deux
   partagent le cadre (le rayon ρ borne les deux) mais un papier qui répond à deux
   questions n'en défend bien aucune.
2. **La validation manque.** Le gain est mesuré à un seul k, une seule taille de bloc,
   deux modèles — et il s'effondre avec RoPE (×1,99 → ×1,23). Publier cela comme résultat
   secondaire d'un papier système serait le publier sans le tester.
3. **La difficulté est ailleurs.** Un partitionnement par contenu casse la contiguïté
   du cache : c'est un problème de **gestion mémoire** (pages, réécriture à chaque
   insertion), pas d'attention. Ce papier-ci a justement pour thèse que ce genre de coût
   système domine le gain algorithmique. Les inclure sans traiter ce coût contredirait
   la thèse du papier dans le papier lui-même.

### État

[FAIT] `17_papier/asp.tex` (9 pages), `refs.bib` (35 entrées, toutes citées et toutes
vérifiées par extraction du titre en première page du PDF), 5 figures générées depuis les
mesures. Compilation propre : 0 référence non résolue, 0 débordement > 5 pt, aucun
placeholder.

[NON FAIT, honnête] Le noyau fusionné du volet 1 n'est pas intégré au modèle du volet 2.
C'est la suite directe, et le papier le dit en section Limites plutôt que de le masquer.


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


### Session 14 — Le sink n'est pas un token special (validation + generalisation)

[QUESTION] Le sink mesure en session 13 etait-il un artefact d'un unique BOS a la position 0 ?

[REPONSE] Non. Trois preuves.
1. SmolLM2-135M n'ajoute AUCUN token special (add_special_tokens=True et False donnent le
   meme token[0]=446 et des sorties identiques). Le sink est donc un token de TEXTE ordinaire.
2. Re-capture independante sur 31 textes locaux du corpus (283 015 caracteres, sans aucun
   token special) : la position 0 porte 30,1 % de la masse, le bloc 0 33,3 %, la fenetre 512
   50,6 %. Sur 4 documents concatenes : 35,6 / 40,3 / 41,5 %.
3. Le correctif (forcer le bloc 0) ramene les tetes catastrophiques a 0,0 % sur les trois
   sequences (7,1 / 13,1 / 26,2 % -> 0,0 %) et l'erreur mediane de 0,072/0,106/0,121 a
   0,048/0,071/0,038 (oracle 0,033/0,049/0,022).

[VALIDATION DU PIPELINE] La replication RoPE a ete verifiee contre les captures existantes
(qk_smol8k.npz) sur les 2048 premieres positions : max|delta K| = 3,906e-03, cosinus =
1,000000 (la RoPE est absolue, donc tronquer la sequence ne change pas ces positions).

[NEW] Le taux d'echec du selecteur par cle moyenne croit avec la longueur du contexte :
7,1 % a 2 048 tokens (wikitext), 26,2 % sur le corpus local a 2 048 tokens, 43-47 % a
8 192 tokens. L'amplitude du sink, elle, est stable (23-36 %).

[FICHIERS] code/capture_local.py ; resultats/capture_local_summary.txt.


### Session 15 — Le selecteur de blocs est pire que le hasard (perplexite)

[HARNAIS] L'attention de SmolLM2-135M est remplacee par un forward propre (fenetre locale W
+ m blocs de Lb choisis, o_proj conserve). Validation : le mode dense patche reproduit la
perte native a l'identique (2,7564 / 2,6260). T=1024, Lb=64, corpus local.

[RESULTATS] dense 2,7564 (ppl 15,74) ; W256 m0 (fenetre seule) 3,7909 (44,30) ;
W256 m4 3,9240 (50,60) ; W256 m8 4,0116 (55,24) ; W256 m4 ALEATOIRE 3,8634 (47,63) ;
W512 m4 4,0235 (55,89).

[LECTURE]
1. La fenetre seule coute +1,0345 nat, soit 67 % de la penalite totale (+1,2558).
2. La degradation est monotone en m : chaque bloc ajoute par le selecteur empire.
3. LE SELECTEUR PAR CLE MOYENNE EST PIRE QUE LE HASARD : 3,9240 contre 3,8634 (graine 1234).
   Il choisit activement les mauvais blocs.
4. Anomalie a verifier : W512_m4 (4,0235) est pire que W256_m4 (3,9240) malgre 768 tokens
   visibles contre 512.

[CONSEQUENCE] La passe 1 par resume compact n'apporte aucune valeur au niveau modele dans
cette configuration ; le forcage du sink ne recuperait que 4 % de la penalite (session 14) ;
et la metrique par tete surestimait l'importance du sink. Le goulot est la SELECTION.

[LIMITES] une tranche de 1024 tokens, un seul modele, un seul tirage aleatoire (la
difference de 0,06 nat n'est pas encore separee du bruit de tirage), V = K.

[FICHIERS] code/perplexite_configs.py ; resultats/perplexite_configs.txt.
