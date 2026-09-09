# Deux mécanismes nouveaux pour la sélection de blocs

**ASP** (allocation séquentielle de précision) et **blocs définis par le contenu**.
Code : [`code/precision_allocation.py`](code/precision_allocation.py),
[`code/asp_pareto.py`](code/asp_pareto.py), [`code/asp_optimizer.py`](code/asp_optimizer.py).
Journal de la découverte : [`../07_journal_recherche.md`](../07_journal_recherche.md).

---

## 1. L'hypothèse implicite que personne ne questionne

[FAIT] NSA ([2502.11089](https://arxiv.org/abs/2502.11089)), Quest, COBS
([2607.09052](https://arxiv.org/abs/2607.09052)), CSA de DeepSeek-V4
([2606.19348](https://arxiv.org/abs/2606.19348)), AsyncTLS
([2604.07815](https://arxiv.org/abs/2604.07815)) et mes propres coresets partagent tous une
même structure, jamais discutée : **le même budget d'octets est alloué à chaque bloc, et
chaque résumé est lu exactement une fois.** Résolution uniforme, passe unique.

[INFÉRENCE] C'est une hypothèse, pas une nécessité. Or le problème sous-jacent — *trouver
les k plus grands parmi n à partir d'estimations bruitées, sous budget de mesure* — est
résolu depuis longtemps dans un autre domaine : l'**identification des meilleurs bras** en
bandits stochastiques (Bubeck–Munos–Stoltz [0802.2655](https://arxiv.org/abs/0802.2655),
Karnin *Sequential Halving*, Carpentier–Locatelli
[1605.09004](https://arxiv.org/abs/1605.09004)). Sa théorie dit que l'allocation optimale
est **séquentielle et non uniforme**.

## 2. Une variante nouvelle du problème de bandit

[HYPOTHÈSE/PROPOSITION] La correspondance n'est pas exacte, et c'est ce qui la rend
intéressante. En bandit classique, tirer un bras `t` fois fait décroître le bruit en
`1/sqrt(t)` par **moyennage stochastique**. Ici, lire `b` octets de résumé fait décroître le
bruit selon une **courbe débit-distorsion déterministe** `sigma(b)` : pas de moyennage,
pas d'aléa, et la précision est bornée par la théorie de la compression.

J'appelle cette classe un **bandit à allocation de précision** : le budget est en *bits par
bras*, et la précision suit une loi débit-distorsion mesurable. À ma connaissance cette
variante n'est pas traitée dans la littérature des bandits ni dans celle de l'attention.

## 3. Formalisation

[DÉRIVATION] Soient `n` blocs candidats, `k` à sélectionner, `l_b` la log-masse vraie du
bloc `b`. Un résumé de `b` octets donne un estimateur de bruit `sigma(b)`. Schéma à deux
passes : `b1` octets pour les `n` blocs, `b2` octets pour les `m` survivants.

**Coût** `B(b1,b2,m) = n·b1 + m·b2`.

**Perte de rappel**, en deux sources indépendantes au premier ordre :

*(1) Élimination à tort en passe 1.* Un bloc de rang vrai `j ≤ k` sort du top-`m` si son
score bruité passe sous celui du bloc de rang `m`. Les deux scores étant bruités
indépendamment, leur différence a pour écart-type `sigma1·sqrt(2)`, d'où

```
L1(b1, m) = somme_{j<=k} w_j · Phi( -Delta_{j,m} / (sigma(b1)·sqrt(2)) )
Delta_{j,m} = l_(j) - l_(m)        w_j = part de masse du rang j
```

*(2) Erreur de classement en passe 2*, donnée par la **courbe de référence** mesurée
(rappel atteignable en fonction du bruit, cf. §5) : `L2(b2) = 1 - R(sigma(b2))`.

**Problème.** `min n·b1 + m·b2` sous `R(sigma(b2)) - L1(b1,m) >= cible`.

Toutes les grandeurs — `sigma(b)`, `Delta_{j,m}`, `w_j`, `R` — sont **mesurables hors ligne**
sur un échantillon de calibration. La configuration optimale se calcule donc sans balayage.

## 4. Algorithme

```
Hors ligne (au prefill), par bloc :
    stocker un résumé grossier  S1(b)  (coreset r1)
    stocker un résumé fin       S2(b)  (coreset r2)

À chaque token décodé, par requête q :
    1. lire S1 pour les n blocs        -> scores grossiers  (n·b1 octets)
    2. garder les m meilleurs          -> survivants
    3. lire S2 pour les m survivants   -> scores fins       (m·b2 octets)
    4. renvoyer le top-k des survivants
```

[INFÉRENCE] Le gain vient de ce que `m << n` : la plupart des blocs sont éliminables à
précision grossière car ils sont loin du seuil de décision. **ASP échange du stockage
contre de la bande passante** — les résumés fins restent en mémoire, on se contente de ne
pas les lire.

**Ce que ce n'est pas.** AsyncTLS raffine la *granularité* (blocs → tokens) à résumé fixe ;
HiSparse ([2608.07009](https://arxiv.org/abs/2608.07009)) hiérarchise le *stockage*
(hôte/GPU). ASP raffine la **précision du résumé à granularité constante**, avec une
allocation pilotée par la distance au seuil.

## 5. Résultats

**Bancs :** Qwen2.5-0.5B et SmolLM2-135M, Q/K réels post-RoPE, 8192 tokens, blocs L=64,
fenêtre locale et puits exclus des candidats, top-8, CPU seul.

### Courbe débit-distorsion mesurée (Qwen, sans RoPE)

| coreset r | vec/bloc | `sigma_disc` | rappel@8 |
|---:|---:|---:|---:|
| 1 | 1,02 | 0,480 | 87,73 % |
| 2 | 2,03 | 0,408 | 89,14 % |
| 4 | 4,07 | 0,356 | 91,02 % |
| 8 | 8,14 | 0,302 | 92,62 % |
| 16 | 16,28 | 0,236 | 95,20 % |
| 32 | 32,6 | — | 97,81 % |

### Gain d'ASP, mesuré par interpolation à rappel égal

[FAIT] Pour chaque configuration ASP, on calcule combien d'octets il faudrait à un schéma
plat pour atteindre **exactement le même rappel** (interpolation du front plat), puis on
prend le rapport. Aucun seuil arbitraire.

| banc | passe 1 | gain médian | meilleur | configs perdantes |
|---|---|---:|---:|---:|
| Qwen, avec RoPE | coreset r=2 | **1,64×** | 2,38× | 1/32 |
| Qwen, sans RoPE | coreset r=1 | 1,55× | 2,25× | 2/40 |
| SmolLM2, avec RoPE | coreset r=4 | 1,36× | 1,87× | 1/24 |
| SmolLM2, sans RoPE | coreset r=4 | 1,29× | 1,66× | 2/24 |

[AUTOCORRECTION] Mon premier calcul annonçait 4,06×. C'était un **artefact de placement de
seuil** (comparaison au premier point plat dépassant une cible discrète, très au-dessus de
celle-ci). Le chiffre correct est entre 1,1× et 2,4×.

[FAIT] **Le filtre de passe 1 gouverne la robustesse, pas le gain.** Passer du mean-pool à
un coreset r=2–4 laisse le gain médian quasi inchangé mais fait tomber les configurations
perdantes de 15/40 à 2/24. **Recommandation : passe 1 = coreset r=2.**

### Le modèle prédictif fonctionne

[DÉRIVATION] En injectant les grandeurs mesurées dans le problème du §3 :

| banc | gain prédit (médian / max) | gain mesuré (médian / max) |
|---|---:|---:|
| Qwen, avec RoPE | 1,47× / 1,80× | 1,64× / 2,38× |
| SmolLM2, avec RoPE | 1,66× / 2,07× | 1,36× / 1,87× |

Le modèle prédit le bon ordre de grandeur et le bon classement des configurations. Il
**sous-estime le rappel absolu** de 2 à 4 points, parce que l'hypothèse d'indépendance
entre les deux sources de perte n'est qu'une approximation au premier ordre.

## 6. Ce que le gain vaut au niveau système

[DÉRIVATION] En réutilisant le modèle roofline construit en session 1 : part de l'indexeur
dans les octets lus par token décodé, et gain total si ce terme est divisé par 1,5.

| modèle | contexte | total lu/token | dont indexeur | part | gain total |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V3.2 | 1M | 7 888 MB | 7 808 MB | **99,0 %** | **1,49×** |
| DeepSeek-V4-Flash | 1M | 423 MB | 320 MB | **75,6 %** | **1,34×** |
| DeepSeek-V4-Pro | 1M | 630 MB | 464 MB | 73,7 % | 1,33× |
| DeepSeek-V4-Flash | 128k | 62 MB | 41 MB | 66,1 % | 1,28× |

[INFÉRENCE] ASP attaque le terme **dominant** du décodage long-contexte. Le décodage étant
borné par la bande passante (intensité arithmétique 131–308 FLOP/octet contre un point de
bascule à ~563), 1,5× sur le sélecteur ≈ **1,34× sur la latence de décodage à 1M**.

## 7. Limites, et comment invalider ASP

1. **Le gain est modeste et variable** (1,1× à 2,4×). Ce n'est pas un changement de régime.
2. **Coût mémoire.** ASP suppose les résumés fins résidents. Pour r=16 sur des blocs de 64
   tokens, cela ajoute ~32 octets/token, soit 15–25 % du KV cache lui-même. Sur un système
   contraint en capacité plutôt qu'en bande passante, l'échange s'inverse.
3. ~~**Coût de branchement** (inconnue principale)~~ — **RÉSOLU, mesuré sur GPU réel.**
   Voir [`../16_gpu/RAPPORT_GATHER.md`](../16_gpu/RAPPORT_GATHER.md). Sur `gfx1152`
   (AMD Radeon 860M, OpenCL) : efficacité médiane **96,2 %**, **0/64 plus lentes**, gather
   à **94,9 %** du débit contigu. **Confirmé sur Tesla T4** (GDDR6) : gather à **99,1 %**
   du contigu. **Et sur A100 PCIe** (HBM2e, 1372 GB/s) : **aucune pénalité mesurable à
   aucune granularité** (ratio min 0,955). La pénalité décroît avec le parallélisme du
   GPU — elle est **nulle en production**. Un second obstacle la remplace toutefois : le
   coût de lancement de la seconde passe (règle 2 ci-dessous).
4. **Modèles génériques, non entraînés en creux.** Qwen2.5-0.5B et SmolLM2-135M sur
   wikitext, contexte 8k. Un modèle entraîné pour le long contexte a une attention plus
   « needle-like », ce qui **augmenterait** probablement le gain d'ASP (plus de blocs
   éliminables tôt) — mais ce n'est pas vérifié.
5. **Deux passes seulement.** La théorie des bandits suggère un nombre de passes
   logarithmique (*successive halving*). Mes essais à trois passes ne battent pas les
   meilleures configurations à deux passes sur ces bancs.

**Deux règles de conception issues des mesures GPU.**

> **(1) Granularité** — `r2 × D × 2 ≥ 2048 octets`. À 1024 o l'efficacité tombe à 50 % sur
> T4 (et le gain réel à 0,99× sur l'iGPU à 512 o). À 4096 o : 85 % ; à 8192 o : 90 %.
>
> **(2) Volume** — dépend du GPU, et c'est devenu la contrainte dominante :
> ~8 Mo sur T4, **~64 Mo sur A100** (22 % d'efficacité sous 16 Mo, 77,6 % au-delà de
> 64 Mo, 94,2 % à 134 Mo). Le coût de lancement d'un noyau (~11,3 µs sur A100) est fixe
> alors que le travail se fait plus vite : plus le GPU est rapide, plus il faut brasser de
> données. **Contrainte invisible sur l'iGPU**, révélée par le T4, dominante sur A100.

Le régime de déploiement réel les satisfait largement (1M de contexte, L=64, r2=16, D=128
→ 15 625 blocs, S2 = 4096 o, 64 Mo de volume). **Seuil d'utilité : ~130 k tokens.**

**Une prédiction réfutée, à consigner.** Le modèle MLP (loi de Little) prédisait un seuil
de 328 o sur T4, plus bas que les 1024 o de l'iGPU. **Mesure : 1024 o sur T4 aussi** — la
calibration est fausse d'un facteur 3. Le seuil s'avère **invariant à ~1 Kio** sur deux
systèmes mémoire très différents, ce qui soutient une explication par granularité **DRAM**
plutôt que par parallélisme mémoire.

**Volet fusion — RÉSOLU (H200).** Le noyau fusionné a été implémenté et mesuré sur
**NVIDIA H200 NVL** (132 SM, sm_90). Conception retenue : **stratification entrelacée sans
aucune synchronisation** — chaque work-group traite les blocs {g, g+G, g+2G, …} et gathère
directement son propre top-q.

| q | fusion / 2-noyaux | gain vs plat | configurations gagnantes |
|---:|---:|---:|---:|
| **8** | **1,57×** | **1,71×** (max 4,58×) | **15/15** |
| 16 | 0,78× | 1,27× | 3/15 |
| 32 | 0,68× | 0,71× | 0/15 |
| 64 | 0,32× | 0,38× | 0/14 |

> **(3) Stratification** — choisir `G ≈ m/8`. Le produit `G·q = m` étant fixé, G gouverne
> le parallélisme et q le travail sériel par équipe ; l'optimum est au quota 1 par strate.
> **Coût mesuré : 2,5 à 3,2 points relatifs de rappel** (mode entrelacé obligatoire — le
> mode contigu coûte jusqu'à 22,3 points).

**Ce qui reste à invalider.** L'intégration dans un moteur de serving réel et la mesure de
bout en bout sur un modèle (volet 2). Le banc reste un microbenchmark de motif d'accès :
il ne modélise ni le calcul des scores d'attention, ni le softmax, ni la pression registre
d'un noyau d'attention complet.

---

## 8. Un second concept, plus ambitieux : **blocs définis par le contenu**

[FAIT] Deuxième hypothèse implicite partagée par NSA, Quest, COBS, CSA, AsyncTLS et
HiSparse : **les blocs sont contigus en position.** La justification est matérielle, jamais
sémantique. Or la masse d'attention n'est pas groupée spatialement (autocorrélation 0,13-0,21
au décalage 1, session 2) : un bloc contigu est un agrégat arbitraire, ce qui gonfle son
rayon `rho` — la quantité même qui borne l'erreur de tout résumé.

[FAIT] Test, à nombre et taille de blocs identiques (L=64, top-8, blocs distants) :

| banc | rho | sigma_disc | **masse absolue captée** |
|---|---:|---:|---:|
| Qwen sans RoPE — position | 12,72 | 0,488 | 21,61 % |
| Qwen sans RoPE — **contenu** | **8,42** (×0,66) | 0,444 | **43,14 %** (**×2,00**) |
| SmolLM2 sans RoPE — **contenu** | 7,90 (×0,67) | 0,459 | **45,20 %** (**×2,36**) |
| Qwen avec RoPE — **contenu** | 13,33 (×0,94) | 0,887 | 54,78 % (×1,23) |
| SmolLM2 avec RoPE — **contenu** | 12,19 (×0,92) | **0,817** (×0,75) | 71,04 % (×1,37) |

[INFÉRENCE] Traduction : il faut ~20 blocs par position pour capter ce que 8 blocs par
contenu captent — soit **~2,5× moins de tokens lus à qualité égale** (sans RoPE).

[FAIT, contre-intuitif] Regrouper sur les clés **pré**-RoPE est **moins bon** que sur les
clés post-RoPE (masse 48,79 % contre 54,78 %). L'attention opérant sur les clés post-RoPE,
c'est cette géométrie-là qu'il faut rendre compacte ; regrouper avant rotation réunit des
clés qui se ré-étalent après.

[INFÉRENCE] Corollaire : **RoPE combat le découpage par contenu** (gain ×2,0-2,4 sans RoPE
contre ×1,2-1,4 avec). C'est le pendant de la découverte NoPE de COBS, vu du côté du
partitionnement. Les architectures à RoPE découplée (MLA, DSA : grand latent sans RoPE +
64 dims RoPE) sont structurellement les mieux placées pour en profiter.

**Limites.** (a) Coût du regroupement : mon k-means équilibré est rédhibitoire à N=1M, il
faudrait un regroupement approché ou en flux. (b) Causalité : un bloc-contenu mélange les
positions, donc c'est une réorganisation **de la phase de décodage**, applicable après le
prefill seulement. (c) Contiguïté mémoire perdue au niveau des positions — mais un cache
paginé l'est déjà. (d) Deux petits modèles, contexte 8k.

**Comment l'invalider.** Mesurer le coût amorti d'un regroupement approché sur 1M de tokens.
Si le prefill s'allonge de plus de ~10 %, le gain de décodage ne compense pas pour les
requêtes courtes.

---

## 9. Toutes les propositions issues de ce travail, classées

| # | Proposition | Ambition | Confiance | Statut |
|---|---|---|---|---|
| **1** | **Blocs définis par le contenu** plutôt que par la position (§8) | **forte** | **élevée** | testé sur 2 modèles × 2 régimes : masse captée ×2,0-2,4 sans RoPE, ×1,2-1,4 avec ; mécanisme (rho) confirmé |
| **2** | **Résumés par coreset** (support) plutôt que par covariance (moments) | moyenne | **élevée** | testé : 2 modèles × 3 tailles de bloc × 2 régimes, gain 4,5× sur COBS à qualité égale |
| **3** | **La structure d'ex æquo** : l'écart entre blocs adjacents (0,07 nat) est 20–37× plus petit que l'incertitude de tout résumé compact | forte (conceptuelle) | **élevée** | mesuré ; explique 4 échecs indépendants et réoriente les priorités du domaine |
| **4** | **Courbe de référence rappel(sigma)** comme outil de mesure de la marge restante | faible | **élevée** | mesurée ; prédit les coresets à ±0,4 pt, échoue de +5 à +8 pt sur COBS/GMM (hétéroscédasticité) |
| **5** | **ASP** — allocation séquentielle de précision | moyenne | **élevée sur le mécanisme, moyenne en production** | gain 1,1–2,4× en octets, **converti à 96 % en temps mural mesuré sur GPU réel** (gfx1152) ; valable sous `r2·D·2 ≥ 1024 o` ; reste à confirmer sur HBM |
| **6** | **Bandit à allocation de précision** comme classe de problème | forte (théorique) | moyenne | formalisé, modèle prédictif validé au premier ordre ; pas de théorème d'optimalité |
| **7** | **Cadre « quantification de mesure sous transformée de Laplace »** unifiant MLA, pooling, éviction et quantification | forte (conceptuelle) | moyenne | argumenté, un théorème prouvé et vérifié ; pas encore utilisé pour comparer les familles entre elles |
| **8** | **Un coreset unique pour les trois branches** de NSA/CSA (compression + sélection + incertitude) | forte | **faible** | non testé ; demande un entraînement |
| **9** | **Sélection certifiée** (séparation-évaluation avec garantie) | forte | **réfutée** | testé et invalidé : `M = s‖q‖ρ ≈ 24 nats`, toute inégalité de concentration est vide |
