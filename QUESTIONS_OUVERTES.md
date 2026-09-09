# Questions ouvertes — classées par (impact × faisabilité)

Chaque question indique ce qui la motive dans ce corpus, un protocole exécutable, et le signal
qui trancherait. Les quatre premières sont testables sans GPU ou sur un GPU unique.

---

## Q1 ⭐⭐⭐ — L'entropie du compresseur CSA est-elle l'hyperparamètre caché de DeepSeek-V4 ?

**Motivation.** Mesuré (SYNTHESE §5) : un compresseur à moyenne uniforme perd 13,6 % (m=4) à
21,8 % (m=16) de masse d'attention par rapport à l'oracle — le « trou de Jensen ». Il se referme
dès β≈0,25–0,5. Le papier V4 ne mentionne jamais l'entropie de `Softmax_row` (éq. 11), alors
qu'elle conditionne la qualité de CSA et que le risque croît avec `m`. Cela pourrait expliquer le
choix conservateur m=4.

**Protocole.**
1. Entraîner deux petits modèles CSA (~100–300 M) identiques, en ne changeant que la
   **température** de `Softmax_row` (ou une pénalité d'entropie sur les poids de compression).
2. Suivre l'entropie moyenne des poids de compression au cours de l'entraînement.
3. Évaluer sur récupération fine (needle-in-a-haystack, MRCR) et sur perplexité longue.
4. Balayer m ∈ {4, 8, 16} pour chaque réglage.

**Signal décisif.** Si la qualité en récupération corrèle avec l'entropie plus qu'avec `m`, alors
la piquosité est le vrai levier — et **m=8 ou 16 devient utilisable**, ce qui doublerait ou
quadruplerait encore l'économie d'indexeur (le terme dominant, cf. SYNTHESE §1).

**Faisabilité.** Élevée : modèles petits, signal mesurable tôt. **Impact.** Très élevé : touche
directement le paramètre le plus coûteux de l'architecture.

---

## Q2 ⭐⭐⭐ — Peut-on réconcilier décodage spéculatif et attention creuse par sélection partagée ?

**Motivation.** Mesuré (SYNTHESE §7) : sous sparsité par token, l'amortissement de la lecture KV
tombe à 41,8 % (γ=8) et l'accélération spéculative s'annule (1,00× à α=0,7). Aucun papier du
corpus ne traite cette interaction, alors que V4 combine CSA et MTP.

**Protocole.**
1. Implémenter trois politiques de sélection pour la vérification de γ+1 candidats :
   (a) indépendante (état de l'art) ; (b) **partagée** — un seul top-k calculé sur la position de
   départ, imposé à tous les candidats ; (c) **union bornée** — union des top-k tronquée à un
   budget fixe.
2. Mesurer conjointement le taux d'acceptation α et les octets lus par token.
3. Comparer l'accélération de bout en bout à γ ∈ {2, 4, 8}.

**Signal décisif.** La politique (b) impose l'amortissement à 100 % au prix d'une baisse de α sur
les positions éloignées. Si la baisse de α est inférieure à ~5 points, le compromis est nettement
gagnant dès γ≥4.

**Faisabilité.** Élevée : implémentable dans vLLM/SGLang sans réentraînement. **Impact.** Élevé :
débloquerait la spéculation sur toute la famille des modèles creux long-contexte.

---

## Q3 ⭐⭐⭐ — Comment quantifier un cache **latent partagé** (MLA, CSA) ?

**Motivation.** Toute la littérature de quantification du KV cache du corpus (TurboQuant, RDKV,
Block-GTQ, INT8/INT4) travaille sur des caches **GQA standard**, où K et V sont distincts. MLA et
CSA stockent un **état latent unique servant à la fois de clé et de valeur** (V4 §2.3.1 : « chaque
entrée compressée sert à la fois de clé et de valeur »). L'erreur de quantification s'y propage
donc **deux fois**, par deux chemins de sensibilité différents (le chemin clé passe par un softmax,
le chemin valeur est linéaire). Aucun papier du corpus n'étudie ce cas.

**Protocole.**
1. Dériver analytiquement la sensibilité de la sortie d'attention à une perturbation de l'entrée
   latente, en séparant contribution « clé » et contribution « valeur ».
2. Vérifier numériquement sur un modèle MLA (DeepSeek-V2-Lite est public et petit).
3. En déduire une allocation de bits **par chemin** et la comparer à une allocation uniforme.

**Signal décisif.** Si les deux chemins ont des sensibilités d'ordres différents, l'allocation
optimale sur cache latent diffère structurellement de celle du cache GQA, et tous les résultats de
quantification publiés sont **non transférables** à MLA/CSA — ce qui concerne toute la famille
DeepSeek.

**Faisabilité.** Moyenne-élevée (analytique + petit modèle). **Impact.** Élevé : comble un angle
mort exact entre deux littératures.

---

## Q4 ⭐⭐ — Le gradient d'énergie en fréquence RoPE permet-il une allocation **statique** ?

**Motivation.** Mesuré (SYNTHESE §6) : au-delà de l'irrégularité par tête que décrit Block-GTQ,
il existe un **gradient systématique en fréquence** — les blocs basse fréquence portent 34,9× plus
d'énergie que les hauts. Si ce gradient est universel, une heuristique statique capturerait
l'essentiel du gain **sans calibration**, ce qui simplifierait radicalement le déploiement.

**Protocole.**
1. Mesurer le profil d'énergie par bloc RoPE sur 8–10 modèles de familles et tailles variées.
2. Ajuster une loi `énergie ≈ f(indice de fréquence)` ; quantifier la part de variance expliquée
   par la fréquence seule, par rapport à la variance par tête.
3. Comparer trois allocateurs à budget égal : uniforme, **statique en fréquence**, Block-GTQ calibré.

**Signal décisif.** Si l'allocation statique atteint >80 % du gain de Block-GTQ, elle le remplace
en pratique (zéro calibration, zéro métadonnée par tête).

**Faisabilité.** Très élevée — mon script `exp_B_rope_bit_allocation.py` fait déjà la mesure sur un
modèle ; il suffit de l'étendre. **Impact.** Moyen-élevé : simplification de déploiement.

---

## Q5 ⭐⭐ — CSA et HCA sont-ils vraiment complémentaires ? (ablation)

**Motivation.** [HYPOTHÈSE] de l'axe 1 : CSA = récupération haute résolution, HCA = résumé
exhaustif basse résolution. L'alternance donnerait les deux régimes. Le papier V4 affirme la
complémentarité mais ne publie pas d'ablation isolant les deux.

**Protocole.** Entraîner à taille égale : CSA-seul, HCA-seul, alterné (V4), et alterné avec ratios
2:1 et 1:2. Évaluer séparément sur (a) récupération fine — needle, MRCR ; (b) agrégation globale —
résumé long, comptage, tri ; (c) perplexité.

**Signal décisif.** Si CSA-seul s'effondre en (b) et HCA-seul en (a), la complémentarité est
établie et le **ratio** devient un levier de conception à part entière, ajustable selon le domaine.

**Faisabilité.** Moyenne (5 entraînements). **Impact.** Élevé pour qui conçoit une architecture.

---

## Q6 ⭐⭐ — Le ratio « KV résident / KV lu » prédit-il le gain d'un système de préchargement ?

**Motivation.** Mesuré (SYNTHESE §3) : V4-Flash a un ratio de 7,8×. LSA (2606.09079), qui
précharge les chunks critiques, annonce 2,8× de débit. [HYPOTHÈSE] : le ratio est une **borne
supérieure** du gain atteignable, atteinte au rappel parfait de l'indexeur prédictif.

**Protocole.** Sur 3–4 architectures creuses de ratios différents, implémenter un préchargeur
simple et mesurer le gain réel. Tracer gain observé contre ratio théorique.

**Signal décisif.** Une relation monotone ferait du ratio une **métrique de conception
standardisable** : une architecture le publierait pour annoncer ce que le serving pourra en tirer.

**Faisabilité.** Moyenne (nécessite un GPU et un vrai système). **Impact.** Moyen-élevé : créerait
un pont métrique entre architecture et serving.

---

## Q7 ⭐⭐ — À quel taux de concurrence V4-Flash bat-il un modèle dense de 13–30 B ?

**Motivation.** [HYPOTHÈSE] de l'axe 4 : le vrai concurrent de V4-Flash n'est pas V3.2 mais un
dense de taille comparable en *calcul actif*. Les poids (135 GB) s'amortissent sur le batch, le
calcul non. Il doit exister un seuil de concurrence.

**Protocole.** Modéliser puis mesurer le coût par token servi (poids lus + KV lu + calcul) en
fonction du batch, pour V4-Flash et pour un dense de 13 B et 30 B, à contextes 32 K / 256 K / 1 M.

**Signal décisif.** Le point de croisement donne une règle de déploiement directement actionnable
(« au-dessus de N requêtes concurrentes, le MoE creux est moins cher »).

**Faisabilité.** Élevée en analytique, moyenne en mesure. **Impact.** Élevé pour l'exploitant.

---

## Q8 ⭐ — Le format FP4 des experts routés dégrade-t-il la qualité, et où ?

**Motivation.** [FAIT] V4 met les experts routés en FP4, ce qui fait passer V4-Flash de ~270 GB à
135 GB et le rend déployable sur un seul B200 (axe 4 §5). C'est l'un des choix les plus
conséquents du papier, et il n'est appuyé par aucune ablation publiée.

**Protocole.** Quantifier en FP4 les experts d'un MoE ouvert (Mixtral, Qwen-MoE), par groupes de
couches, et mesurer la dégradation par capacité (raisonnement, code, multilingue, contexte long).

**Signal décisif.** Si la dégradation se concentre sur certaines couches ou capacités, une
allocation FP4/FP8 mixte par couche serait un gain immédiat.

**Faisabilité.** Élevée. **Impact.** Moyen.

---

## Q9 ⭐ — Comment l'attention creuse interagit-elle avec le *chunked prefill* ?

**Motivation.** Sarathi-Serve découpe le prefill en chunks pour éviter les stalls. Sous attention
creuse, un chunk de requêtes doit sélectionner dans les chunks déjà traités — la sélection devient
incrémentale et le coût de l'indexeur se répartit différemment. Aucun papier du corpus ne croise
les deux.

**Protocole.** Modéliser le coût de prefill chunké sous CSA en fonction de la taille de chunk, puis
mesurer. Chercher la taille optimale, qui diffère probablement de l'optimum dense.

**Faisabilité.** Moyenne. **Impact.** Moyen.

---

## Q10 ⭐ — Le routage multi-modèles peut-il exploiter la longueur de contexte comme signal ?

**Motivation.** Le survey du routage (2603.04445) route sur la difficulté de la requête. Or mes
mesures montrent que le coût varie de 28 à 131 GFLOPs/token (V4-Flash) **selon le contexte seul**
— un facteur 4,7 indépendant de la difficulté. Un routeur ignorant le contexte laisse cela sur la
table.

**Protocole.** Ajouter la longueur de contexte comme feature d'un routeur cascade, et mesurer le
coût total à qualité constante par rapport à un routeur difficulté-seule.

**Faisabilité.** Élevée. **Impact.** Moyen.

---

## Récapitulatif

| # | Question | Impact | Faisabilité | Sans GPU ? |
|---|---|---|---|---|
| Q1 | Entropie du compresseur CSA | ⭐⭐⭐ | élevée | non (entraînement) |
| Q2 | Sélection partagée pour la spéculation | ⭐⭐⭐ | élevée | non (1 GPU) |
| Q3 | Quantification du cache latent partagé | ⭐⭐⭐ | moyenne-élevée | **oui (partiel)** |
| Q4 | Allocation statique en fréquence RoPE | ⭐⭐ | très élevée | **oui** |
| Q5 | Ablation CSA vs HCA | ⭐⭐ | moyenne | non |
| Q6 | Ratio résident/lu comme prédicteur | ⭐⭐ | moyenne | non |
| Q7 | Seuil de concurrence MoE vs dense | ⭐⭐ | élevée | **oui (analytique)** |
| Q8 | FP4 sur experts routés | ⭐ | élevée | non |
| Q9 | Sparsité × chunked prefill | ⭐ | moyenne | **oui (modèle)** |
| Q10 | Routage sensible au contexte | ⭐ | élevée | **oui (partiel)** |

---

# Série T — questions orientées systèmes & économie (v2, vérifiées)

Les dix questions T complètent la série Q (orientation modèle) par l’angle serving/économie. Classement [INFÉRENCE — jugement d’expert] par impact potentiel × faisabilité :

| # | Thème | Impact | Faisabilité |
|---|-------|--------|-------------|
| T3 | Goulot de transfert KV en P/D désagrégé | très élevé | moyenne |
| T9 | Vrai coût $/M tokens de V4-Flash à 1M contexte | très élevé | faible (données privées) |
| T5 | Erreurs INT4 × CSA : multiplicatives ? | élevé | élevée (mesurable) |
| T4 | Frontière de batch du décodage spéculatif | élevé | élevée |
| T8 | Surface d’attaque Mistletoe et mitigations | élevé | moyenne |
| T6 | Cache de préfixe × attention creuse | élevé | moyenne |
| T2 | Optimum de granularité token/bloc | moyen | moyenne |
| T1 | Loi d’échelle des indexeurs | élevé | faible |
| T7 | Point de bascule TaiChi multi-SLO | moyen | moyenne |
| T10 | La distorsion à 128K prédit-elle celle à 1M ? | moyen | élevée |

## T1. Faut-il une loi d'échelle pour les indexers d'attention creuse ?
**Impact 5 · Faisabilité 4** · Axe 1
Constat : EAGLE-3 démontre que le speedup du draft spéculatif suit une scaling law en données d'entraînement (8× données → +1.4×) [2503.01840]. DSA entraîne son indexer sur 2.1B tokens (warm-up) ; LSA entraîne le sien en ~1 GPU-hour [2606.09079]. Personne n'a mesuré la courbe indexer-accuracy/qualité vs volume d'entraînement.
**Protocole** : sur V4-Flash open-weights, entraîner des indexers LSA à budgets {0.1, 1, 10, 100} GPU-hours ; mesurer (recall@r vs golden attention, LongBench-v2, MRCR) ; ajuster une loi puissance ; vérifier si le plafond MRCR (48% vs 76%) se lève avec 10× plus de données — c'est le test falsifiable du diagnostic des auteurs LSA (capacité insuffisante du dual-encoder standalone).


## T2. La granularité de sélection (token vs bloc) a-t-elle un optimum dépendant de L ?
**Impact 5 · Faisabilité 4** · Axe 1
Constat : DSA défend top-k=2048 tokens [2512.02556] ; V4 revient aux blocs m=4/m'=128 [2606.19348] ; DashAttention critique les budgets fixes [2605.18753]. La lignée DeepSeek a fait les deux, sans publier la comparaison.
**Protocole** : même backbone, trois heads de sélection (token-level k=2048 ; block m=4 ; adaptatif DashAttention) entraînées avec la même recette (continued-pretraining ~10B tokens, pas 1T, à échelle 8B pour coût) ; mesurer LongBench-v2/MRCR/RULER à {32K, 128K, 512K, 1M} ; rapporter qualité/FLOPs/latence. Le résultat trancherait si le coût du indexer L·k devient dominant à 1M — la raison plausible du retour aux blocs.


## T3. Le transfert KV inter-nœuds est-il le goulot de la désagrégation à 1M tokens ?
**Impact 4 · Faisabilité 4** · Axe 4
Constat : V4 consacre sa section §3.5.2 aux trois stratégies de placement (full/checkpoint/zero) car le SWA est ~8× plus volumineux que le CSA/HCA compressé [2606.19348] ; LSA obtient 2.7× concurrency avec un transfert sélectif [2606.09079 §4] ; Mooncake chiffre le cache mais pas le réseau [2407.00079].
**Protocole** : cluster 8×H20 + RDMA 200 Gb/s ; servir V4-Flash à {128K, 512K, 1M} ; mesurer goodput vs bande passante allouée au transfert {25, 50, 100}% ; comparer les 3 stratégies de V4 §3.5.2 ; si le goodput plafonne même à 100% RDMA, la désagrégation on-disk est morte au-delà d'un seuil de longueur — résultat structurant pour la topologie des clusters.


## T4. Le décodage spéculatif a-t-il une frontière de batch prévisible (modèle analytique + mesure) ?
**Impact 4 · Faisabilité 5** · Axe 3
Constat : EAGLE-3 : 3.0-6.5× dans le harnais (Table 1) → ×2.36 réel (SGLang bs=1, Table 4 : 373.25 vs 158.34 t/s) → 1.81× (b=2) → 1.32× (b=32) → 1.38× (b=64, Table 3, H100) ; EAGLE-1 négatif dès b=24 (0.93×) jusqu'à 0.88× (b=48) ; sur vLLM/A100 EAGLE-3 tombe à 1.01× à b=56 (Table 5). Deux tensions emboîtées : harnais vs engine (facteur ~2.3), memory-bound vs compute-bound (facteur ~1.4-4.4 selon b). La transition est qualitative ; aucun papier ne donne la formule. [2503.01840]
**Protocole** : (a) modèle roofline : speedup(B) = f(acceptance τ, draft cost, batch B, HBM BW, FLOPs peak) — dériver le batch de bascule B* ; (b) valider sur V4-Flash + EAGLE-3 sur 8×H20 en sweep B∈{1..128} ; (c) publier B* par (modèle, GPU) comme table de déploiement. Falsifiable : si B* mesuré ≠ prédit à >30%, le modèle roofline est faux.


## T5. Quantization INT4 × compression CSA : les erreurs se multiplient-elles ?
**Impact 4 · Faisabilité 5** · Axe 1 × 2
Constat : V4 garde FP8 pour le KV hors-RoPE et BF16 pour les 64 dims RoPE [2606.19348 §2.3.4] ; INT4 domine le rang à budget égal [2604.11501] ; TurboQuant atteint ≥4.5× near-lossless [2504.19874]. Personne n'a quantisé un cache déjà compressé par blocs appris.
**Protocole** : V4-Flash checkpoint ; quantizer le KV compressé à {INT8, INT4, INT3 RoPE-aware} ; mesurer LongBench-v2 + MRCR à 512K ; si la distorsion est super-additive (INT4 sur cache compressé ≠ INT4 sur cache dense), c'est un résultat négatif important pour la combinabilité des techniques — le dogme « on empile les compressions » est actuellement non vérifié.


## T6. Un cache de préfixe (Mooncake) fonctionne-t-il avec une attention qui ne lit que k blocs ?
**Impact 4 · Faisabilité 3** · Axe 4 × 1
Constat : Mooncake gagne +525% via la réutilisation de préfixes [2407.00079] ; CSA/HCA ne lit que k=512/1024 blocs par query [2606.19348] — le « préfixe » utile n'est pas le KV complet mais les indexers/sélections.
**Protocole** : simuler des charges agentiques (réutilisation 60%) ; mesurer le hit-ratio d'un cache indexant (a) le KV brut, (b) les indexers CSA, (c) les blocs sélectionnés ; si (b)/(c) dominent, le design des serving stacks doit changer : le cache devient un cache de sélections. Résultat de co-design système×architecture.


## T7. Le point-slop de TaiChi est-il stable sous des SLO multi-classes ?
**Impact 3 · Faisabilité 4** · Axe 4
Constat : TaiChi : agrégation si TTFT serré, désagrégation si TPOT serré, ni l'un ni l'autre aux points équilibrés ; +77% goodput en switchant [2508.01989].
**Protocole** : étendre TaiChi à K classes de SLO (contrats réels) ; problème d'allocation multi-classe (bin-packing) ; comparer au switch global. Si le point-slop éclate en K régions, la topologie de cluster doit être par classe — sinon le switch global suffit.


## T8. L'attaque Mistletoe : quelle surface exacte et quelles mitigations ?
**Impact 3 · Faisabilité 4** · Axe 3 (sécurité)
Constat : 06_ressources_en_ligne.md signale Mistletoe [2605.14005] contre les déploiements spéculatifs partagés ; le papier n'est pas dans le corpus — le mécanisme est non vérifié ici.
**Protocole** : (1) obtenir le papier (LACUNES) ; (2) reproduire sur SGLang isolé : l'attaque exploite présumément l'écart draft/cible (si l'attaquant peut faire diverger draft et cible, il force des rejets coûteux ou des états internes observables) ; (3) mesurer l'overhead de vérification supplémentaire (side-channel timing) ; (4) tester les mitigations (vérification en aveugle, isolation par tenant, taux de rejet limité). Prioritaire pour tout déploiement multi-tenant spéculatif.


## T9. Quel est le vrai $/M tokens de V4-Flash à 1M de contexte ?
**Impact 3 · Faisabilité 3** · Axe 5
Constat : V3.2 publie 0.2 $/M (prefill) et 0.8 $/M (decode) à 128K [2512.02556 Fig.3] ; V4 publie des ratios FLOPs/KV (10%/7%) à 1M mais pas de $ [2606.19348]. La conversion FLOPs→$ est invalide en memory-bound.
**Protocole** : construire un estimateur de coût paramétrique (poids actifs FP4/FP8, KV compressé, bande passante HBM, prix $/h) ; l'étalonner sur les points V3.2 publiés (0.2/0.8 $/M @128K) ; prédire le V4-Flash à {128K, 512K, 1M} ; confronter aux prix publics DeepSeek (api-docs.deepseek.com) — la prédiction est falsifiable publiquement.


## T10. La distorsion de quantization à 128K prédit-elle la distorsion à 1M ?
**Impact 3 · Faisabilité 5** · Axe 2
Constat : TurboQuant/Block-GTQ valident à 32K-128K [2504.19874, 2606.24033] ; les modèles à 1M tokens sont le nouveau régime. RDKV suggère que l'attention se concentre sur peu de tokens (2.48% rétention suffit) — mais la part « queue » de la distribution est précisément ce que la quantization à petit budget détruit en premier.
**Protocole** : mesurer la TV-distance entre attention quantisée et dense à {8K, 32K, 128K, 512K, 1M} pour {FP8, INT4, INT3} ; si la distorsion croît super-linéairement en L, les benchmarks courts sur-estiment la qualité des caches quantisés longs — résultat négatif transversal qui contredirait la pratique standard d'évaluation.

---

