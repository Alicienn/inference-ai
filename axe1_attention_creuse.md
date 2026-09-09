# Axe 1 — Attention creuse et contexte long : NSA → DSA → CSA/HCA → LSA

## 1. La lignée, et ce que chaque étape corrige

| Étape | Papier | Idée | Ce qu'elle corrige | Ce qu'elle laisse ouvert |
|---|---|---|---|---|
| **NSA** | [2502.11089](https://arxiv.org/abs/2502.11089) | Attention creuse *native* (entraînée, pas post-hoc), alignée matériel | La sparsité post-hoc dégrade la qualité et n'accélère pas vraiment | Sélection encore coûteuse |
| **DSA** | [2512.02556](https://arxiv.org/abs/2512.02556) | *Lightning indexer* + top-k fin par token ; O(L²)→O(Lk) | Rend la sparsité fine-grain praticable à l'échelle | **L'indexeur reste en O(n)** |
| **CSA/HCA** | [2606.19348](https://arxiv.org/abs/2606.19348) | Compresser (m=4 / m'=128) *puis* sélectionner | Divise le terme O(n) de l'indexeur par m | Entropie du compresseur (§4) |
| **LSA** | [2606.09079](https://arxiv.org/abs/2606.09079) | *Neural Memory Indexer* prédictif ; ne garde en HBM que les chunks critiques | La mémoire résidente, que la sparsité ne réduisait pas | Latence de préchargement |

**[INFÉRENCE]** Cette lignée n'est pas une suite d'améliorations incrémentales du même objet :
c'est une **remontée successive du goulot d'étranglement**. NSA règle la qualité, DSA règle le
calcul d'attention, CSA règle le coût de la sélection, LSA règle l'empreinte mémoire. À chaque
étape, le terme dominant précédent est traité et un nouveau apparaît.

## 2. Le point aveugle de DSA, mesuré

**[FAIT]** Reconstruction indépendante (voir [`METHODES.md`](METHODES.md)) : à 1M de contexte,
DeepSeek-V3.2 consomme **1 108 GFLOPs par token décodé, dont 999 (90,2 %) pour le seul indexeur**.
Le calcul d'attention centrale, celui que la sparsité optimise, n'en représente que 3,1 %.

L'indexeur DSA score la requête contre les `n` clés d'indexation : coût `n_I × c_I × n × 2` FLOPs
par couche, soit 16,4 GFLOPs/couche à n=1M, × 61 couches = 1,0 TFLOP. En octets, il doit **lire
7,9 GB par token décodé** (61 couches × 1M × 128 octets).

**[INFÉRENCE]** C'est le paradoxe de l'attention creuse fine-grain : le mécanisme de sélection est
dense par construction. On ne peut pas savoir quels `k` éléments sont pertinents sans regarder les
`n`. La sparsité déplace le coût du calcul vers la sélection sans le supprimer.

## 3. La réponse de V4 : compresser d'abord

**CSA** (Compressed Sparse Attention, §2.3.1 du papier V4) :
1. compresse les entrées KV de chaque `m=4` tokens en une entrée, par **softmax pondéré appris**
   sur `2m` éléments avec biais positionnels apprenables (éq. 11-12) ;
2. applique DSA sur les `n/4` blocs compressés, top-k = 512 ;
3. attention MQA à KV partagé (l'entrée sert de clé *et* de valeur), dim 512 ;
4. branche complémentaire de fenêtre glissante (`n_win`=128) + *attention sink*.

**HCA** (Heavily Compressed Attention, §2.3.2) : même compression mais `m'=128`, **sans
chevauchement et sans sélection** — attention dense sur `n/128` entrées. Les deux alternent.

**[INFÉRENCE]** CSA et HCA ne sont pas deux variantes du même compromis mais **deux fonctions
complémentaires** :
- **CSA = récupération haute résolution.** m=4 préserve le détail, la sélection top-k concentre le
  budget. C'est le mécanisme « aiguille dans une botte de foin ».
- **HCA = résumé exhaustif basse résolution.** m'=128 est trop grossier pour récupérer un détail,
  mais il *voit tout le contexte* sans exception et coûte `n/128` entrées. C'est le mécanisme
  « de quoi parle globalement ce document ».

L'alternance donne au modèle les deux régimes à chaque paire de couches. **[HYPOTHÈSE]** Un
modèle CSA-seul échouerait sur les tâches d'agrégation globale (résumé, comptage) ; un modèle
HCA-seul échouerait sur la récupération fine (MRCR, needle-in-haystack). Testable par ablation.

## 4. Ce que mes mesures ajoutent

Voir [`SYNTHESE.md`](SYNTHESE.md) §4 et §5 pour le détail. En résumé :

**[FAIT]** La justification intuitive de la compression par blocs (« les tokens voisins sont
pertinents ensemble ») est **fausse** : autocorrélation de la masse d'attention = 0,207 au
décalage 1, 30,4 % d'aiguilles isolées.

**[INFÉRENCE]** Le vrai mécanisme est asymétrique : la compression divise le terme **croissant**
(indexeur, O(n/m)) et ne dégrade qu'un terme **constant et borné** (dilution du budget top-k). À
budget d'indexeur égal, la sélection par blocs récupère 7 à 21 % de masse d'attention en plus.

**[INFÉRENCE]** Le risque résiduel est le **trou de Jensen** : scorer par clé compressée
sous-estime les blocs à aiguille. Perte mesurée jusqu'à −21,8 % (m=16) si le compresseur fait une
moyenne uniforme ; refermé dès β≈0,5. L'entropie de `Softmax_row` est donc l'hyperparamètre
critique caché de CSA. **Non discuté dans le papier.**

**[INFÉRENCE]** m=4 paraît conservateur au vu de mes mesures (m=8 fait mieux à budget égal). Deux
explications compatibles : (a) le trou de Jensen croît avec m, et m=4 borne le risque si le
compresseur n'est pas parfaitement piqué ; (b) des tâches de récupération fine, absentes de mon
protocole, pénalisent m élevé. Question ouverte Q1.

## 5. LSA : la conséquence système du découplage résident/lu

**[FAIT]** Le papier LSA part du constat que « les mécanismes d'attention creuse modernes réduisent
les FLOPs par étape de décodage à un niveau quasi constant, mais l'empreinte mémoire du KV cache
reste entière » — la mémoire est « gaspillée sur du contexte inactif ». Il annonce 2,8× de débit
et 2,7× de concurrence à 1M.

**[INFÉRENCE]** C'est exactement le découplage que je mesure en [`SYNTHESE.md`](SYNTHESE.md) §3 :
V4-Flash a 3,07 GB de cache résident mais n'en lit que 423 MB par token (ratio 7,8×). Si seuls
423 MB sont touchés, garder 3,07 GB en HBM est du gaspillage — d'où l'idée de LSA de déporter et
de **précharger** les chunks critiques. Mon analyse **prédit** l'existence de LSA : la métrique
« octets lus / octets résidents » est le rendement qu'un système de préchargement peut capturer.

**[HYPOTHÈSE]** Le ratio résident/lu (7,8× pour V4-Flash) est une **borne supérieure du gain
atteignable** par un système type LSA, atteinte seulement si l'indexeur prédictif a un rappel
parfait. Les 2,8× annoncés par LSA sont cohérents avec ce plafond de 7,8× moyennant un rappel
imparfait et un surcoût de transfert. **Ce ratio devrait être rapporté comme métrique standard**
des architectures creuses : il annonce ce que le serving pourra en tirer.

## 6. Limites de mes mesures

- Attention réelle mesurée sur **GPT-2** (117M, contexte 1024, attention dense apprise) : les
  valeurs absolues ne se transposent pas à un modèle 1M-contexte entraîné *nativement* creux, dont
  les distributions d'attention sont façonnées par la sparsité elle-même. La **structure**
  (concentration extrême, aiguilles isolées, absence de regroupement spatial) est en revanche un
  phénomène documenté comme universel.
  **Répliqué sur SmolLM2-135M** (RoPE, GQA — [`SYNTHESE.md`](SYNTHESE.md) §9) : mêmes conclusions,
  avec un regroupement spatial encore plus faible (ρ=0,129) et 46,6 % d'aiguilles isolées. Le gain
  à bande passante égale y est cependant plus modeste (+6,1 % contre +9,8 % à m=4) : les
  amplitudes dépendent du modèle, les classements non.
- Mes sélections par blocs utilisent un **score oracle** (somme des attentions réelles), sauf en
  §5 où je simule explicitement le scoring par clé compressée. Les résultats §4 sont donc des
  **bornes supérieures** de ce que CSA atteint réellement.
- Aucune mesure de qualité en aval (perplexité, benchmarks) : je mesure de la masse d'attention
  récupérée, un proxy.


## Addenda v2 — garde-fous et corrections (re-vérifiés à la source)

**[FAIT]** LSA n’est pas gratuit : sur MRCR, l’attention restreinte aux seuls blocs retenus fait chuter la précision de 76,0 % à 48,0 % (« severe breakdown ») [2606.09079, §3.3.2]. Fournir à l’indexeur 50 % des blocs « dorés » (pipeline golden par vote majoritaire inter-couches) réduit l’écart à ~2 points près — l’indexeur seul ne retrouve pas tous les blocs critiques.

**[FAIT]** Fuite volumique du gater : ratio de rétention descendant à 8,4 %, mais volume absolu de blocs retenus ×2,5 — la gate ponctuelle (Sigmoid) ne contrôle pas le volume total [2606.09079, ablations].

**[FAIT]** Recette d’entraînement V3.2/DSA : 943,7B tokens sur 15 000 steps [2512.02556]. *(Correction v1 : attribuée à tort à V4.)*

**[FAIT]** Mémoire MLA (V3) : par token et par couche, le cache stocke le latent compressé c_KV (512 dims) et la clé de décompression k_R (64 dims) [2412.19437, §2.1]. **[INFÉRENCE]** Total 576 dims/token/couche, soit ≈ 68,6 Kio/token sur 61 couches (576 × 2 octets × 61) — le « ~70 Ko/token » de la v1 était notre arithmétique, pas une citation.

**Compléments vérifiés repris de la note v1 :**

- **Lignée : V3 (MLA) → NSA → DSA (V3.2) → CSA/HCA (V4) → LSA**
- Date : 2026-08-30 · Corpus : 31 papiers arXiv (extraction pdftotext -layout, vérifiée section par section)
- [FAIT] NSA [2502.11089] entraîne l'attention creuse **nativement** (from scratch, 270B tokens sur 27B-MoE), avec 3 branches parallèles :
- - **Compression** : blocs de l=32 tokens, stride 16, MLP de compression ;
- - **Sélection** : top-n=16 blocs de l'=64 tokens, agrégation intra-groupe GQA ;
- - **Fenêtre glissante** : w=512 tokens ;
- Ses gains : 9.0× forward, 6.0× backward, **11.6× attendu au décodage** à 64k (Table 4, H100) [FAIT].
- Sa contribution méthodologique majeure est double (§2.1-2.2) :
- - **CSA** (Compressed Sparse Attention) : compression apprise m→1 (m=4 blocs de tokens → 1 entrée KV compressée via softmax de poids), puis sélection top-k=512 (V4-Flash) / 1024 (V4-Pro) **parmi les blocs compressés** via un indexer "Lightning" (64 têtes, c_I=128) ;
- - **HCA** (Heavily Compressed Attention) : compression m'=128→1 — un KV compressé tous les 128 tokens, sur lequel on fait de l'attention **dense** (comme une summary-tape) ;
- - CSA et HCA entrelacées couche par couche (premières couches HCA pour la profondeur, couches suivantes CSA/HCA interleaved), SWA n_win=128 en branche additionnelle, attention sink appris, RMSNorm des queries/KV, RoPE partiel 64 dims avec RoPE(-i) sur les sorties (position relative sur la sortie aussi).
- Les trois leviers de précision complètent : KV mixte BF16(RoPE dims)/FP8(reste) ≈ /2 ; indexer FP4 ; experts MoE FP4.
- [FAIT] LSA [2606.09079] (Tencent, projet suspendu) n'est pas une architecture d'entraînement mais un **plugin d'inférence** : un Neural Memory Indexer (dual-encoder, sigmoid, seuil 0.5, déclenché tous les τ=64 steps) prédit les chunks KV dont le futur aura besoin, et ne charge en GPU que ceux-là. Entraînement backbone-free (~1 GPU-hour H20), focal loss γ=2, négatifs 3:1, r=2048. Résultats : KV GPU = 13.5% de la baseline en moyenne, +0.6% acc moyenne, 0.30× compute/token à 1M, 2.8× throughput, 2.7× concurrency sur 8×H20. MAIS (§3.3, honnête) : échec MRCR 76→48% (dépendance dense globale que l'indexer ne peut pas satisfaire), plafond de généralisation 2× la longueur d'entraînement (entraîné 512K ⇒ hypothèse de déclin >1M), fuite marginale du gater sigmoid (rétention 8.4% mais volume absolu ×2.5 de 125K→500K).
- | NSA | 2502.11089 | 2025 | blocs (l=32/l'=64) + fenêtre 512 | — | O(L·n·l') | 11.6× décodage attendu @64k | sélection par blocs, 3 branches | H100 |
- | DSA (V3.2) | 2512.02556 | 2025 | tokens (top-k=2048) | ~2% baseline GQA8/1M (V4) | O(L·k) | 3× coûts $/M (0.7→0.2 prefill, 2.4→0.8 decode @128K) | top-k fixe ; coûteux en très long | H800 |
- | CSA/HCA (V4-Flash) | 2606.19348 | 2026 | blocs m=4/m'=128 + SWA 128 | **7% de V3.2 @1M** ; ~2% GQA8-BF16 | **10% de V3.2 @1M** | 10× réduction FLOPs vs V3.2 | confond aussi l'effet taille (13B vs 37B actifs) | H800/H100 |
- | CSA/HCA (V4-Pro) | 2606.19348 | 2026 | idem, k=1024 | 10% de V3.2 @1M | 27% de V3.2 @1M | ~4× FLOPs vs V3.2 | comparaison architecturalement propre | H800/H100 |
- | LSA (sur V4-Flash) | 2606.09079 | 2026 | chunks futurs prédits | 13.5% GPU KV vs baseline | 0.30× vs V4-Flash | 2.8× throughput, 2.7× concurrency | échec MRCR ; plafond 2× train-len ; overhead constant | H20 |
- **Remarque méthodologique critique** : le "10% FLOPs / 7% KV" de V4-Flash vs V3.2 mélange deux effets — architecture (CSA/HCA + précision) **et** taille du modèle (13B actifs vs 37B). La comparaison architecturalement propre est V4-Pro (49B actifs ≈ 37B de V3.2) : **27% FLOPs / 10% KV** [arXiv 2606.19348, §1]. Un modèle plus petit a mécaniquement moins de FLOPs/token à longueur fixe. Pour un praticien : le gain architecture-seule est ~3.7× (FLOPs) et ~10× (KV), pas 10×/14×. [INFÉRENCE]
- 3. **Top-k fixe (NSA/DSA) vs adaptatif (DashAttention)** : DashAttention [2605.18753] attaque frontalement les budgets de sparsité fixes : « fixed sparsity budgets allocate disproportionate compute to over-attended and under-attended regions ». 3.36× vs FlashAttention-3 à 96K contre les ~2-3× de NSA — mais sur des configs différentes (modèle et longueur diffèrent). Comparaison directe manquante. [FAIT pour les revendications]
- 4. **LSA échoue là où CSA/HCA+re-lecture dense réussissent (MRCR)** : LSA garde les HCA layers « completely global » (retient Top 50%/25%/10% des chunks CSA) — et pourtant échoue à MRCR (48%) quand V4-Flash standard réussit (76%). Diagnostic des auteurs : l'indexer standalone ne peut pas faire de la récupération dense globale. Contradiction apparente avec le slogan « LSA garde l'équivalent du contexte » — la capacité dépend de la tâche. [FAIT]
- 5. **V4-Flash = 10% FLOPs de V3.2 « à 1M tokens »** : le rapport s'améliore avec la longueur (à 32K, V4-Flash ≈ 3.2/4.5×10^9 vs V3.2 32.2×10^9 FLOPs/token) mais la valeur "10%" est spécifique au point 1M. Généraliser sans condition de longueur est trompeur. [FAIT pour la spécification de longueur]
- - **Sélection par blocs + résumé dense (CSA+HCA)** : motif transférable à tout Transformer (NSA a déjà montré la faisabilité from scratch sur 27B).
- - **Indexer "Lightning" FP4** : transférable mais dépend du budget d'entraînement de l'indexer ; la recette V3.2 (indexer-first warm-up, KL vs agrégée L1, 2.1B tokens) est le guide.
- - **RoPE partiel 64 dims + RoPE(-i) sur sortie** : transférable à toute attention compressée (résout le problème du position embedding absolu dans la sortie pondérée).
- - **mHC** : dépend de la reformation du flux résiduel — exige l'entraînement from scratch (V4 l'a fait sur 32-33T tokens).
- 3. La dépendance dense globale (MRCR) est-elle satisfaisable par une couche HCA unique (summary-tape dense à m'=128), ou faut-il plusieurs échelles ?
- 4. L'interaction LSA × désagrégation P/D (§4 de LSA) est prometteuse (2.7× concurrency) mais non explorée systématiquement.
- 5. L'échelle d'entraînement des indexers (2.1B tokens pour DSA, 1 GPU-hour pour LSA) — quelle loi d'échelle ? Première donnée : EAGLE-3 montre speedup ∝ data pour les drafts ; analogie à tester pour les indexers.

## 7. Références du corpus

- `01_deepseek/DeepSeek-V4_Million-Token-Context_2606.19348.pdf` — §2.3 (CSA/HCA), §4.2.1 (config)
- `01_deepseek/FlashMemory-DeepSeek-V4_Lookahead-Sparse-Attention_2606.09079.pdf` — LSA
- `01_deepseek/DeepSeek-V3.2_DSA_2512.02556.pdf` — DSA, top-k=2048
- `01_deepseek/NSA_Native-Sparse-Attention_2502.11089.pdf`
- `03_attention_kv_cache/SpargeAttention2_2602.13515.pdf`, `DashAttention_2605.18753.pdf` — sparsité entraînable, alternatives
- `03_attention_kv_cache/FlashAttention-3_2407.08608.pdf` — kernel de référence
