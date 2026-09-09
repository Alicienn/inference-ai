# Axe 3 — Décodage spéculatif, et son interaction non documentée avec l'attention creuse

## 1. État de l'art : EAGLE-3 comme standard de fait

**[FAIT]** EAGLE-3 ([2503.01840](https://arxiv.org/abs/2503.01840)) abandonne la contrainte de
prédiction de *features* d'EAGLE-2 et fusionne des features **multi-niveaux** du modèle cible
(bas, moyen, haut) pour alimenter une tête de brouillon légère. Accélérations rapportées : « approximately 3.0x-6.5x » vs décodage autorégressif naïf (abstract). Table 1 (5 tâches × 4 cibles, batch 1, greedy) : moyennes EAGLE-3 — Vicuna-13B 5,51× ; Llama-3.1-8B 4,44× ; Llama-3.3-70B 4,12× ; DeepSeek-R1-Distill-Llama-8B 4,16× ; extrêmes 6,47× (V13B, HumanEval) à 3,08× (DSL, CNN/DM). Intégré à vLLM, SGLang et TensorRT-LLM.

**[FAIT]** La réalité en moteur chargé est bien plus rude (même papier) : SGLang bs=1, H100, Llama-3.1-8B, MT-bench : 158,34 → 373,25 tokens/s, soit ×2,36 (EAGLE-2 : 244,10, ×1,54) — contre ×5,51 dans le harnais batch-1 du même papier : écart ×2,3 d'overheads d'engine. En batch croissant (SGLang v0.4.4, chaîne=3) : 1,81× (b=2) → 1,62× (b=8) → 1,32× (b=32) → ~1,3-1,4× (b=64) ; EAGLE-1 passe sous 1× dès b=24 (0,93× ; minimum 0,88× à b=48) ; sur vLLM/A100 (chaîne=2), EAGLE-3 tombe à 1,01× à b=56. Le spéculatif est un outil de latence (faible batch), pas de throughput.

**[FAIT]** Différence structurelle avec Medusa : Medusa attache des têtes parallèles indépendantes
(chacune prédit à distance 1, 2, 3…) qui **ne partagent pas de contexte** entre elles ; EAGLE-3
utilise une tête autorégressive sur les features fusionnées, ce qui la rend plus stable quand la
longueur de spéculation augmente.

**[FAIT]** Le corpus contient aussi : vérification de préfixes en parallèle
([2605.04263](https://arxiv.org/abs/2605.04263)), décodage spéculatif par cross-attention
([2505.24544](https://arxiv.org/abs/2505.24544)), et EAGLE-Pangu pour NPU Ascend
([2603.08088](https://arxiv.org/abs/2603.08088)) — utile comme point de vue hors-GPU.

## 2. Le modèle de coût standard, et ce qu'il oublie

Le modèle usuel (Leviathan et al.) : avec un taux d'acceptation α et une longueur de brouillon γ,
le nombre attendu de tokens acceptés par passe vaut

```
E(α, γ) = (1 − α^(γ+1)) / (1 − α)
```

et l'accélération vaut `E / (γ·c_draft + c_verify)`, avec `c_verify = 1` puisque la vérification
des γ+1 candidats se fait en une passe.

**[INFÉRENCE] Ce que ce modèle sous-estime.** Poser `c_verify = 1` revient à dire que vérifier
γ+1 tokens coûte autant qu'en décoder un. C'est vrai en régime *compute-bound*. Or en contexte
long le décodage est **borné par la bande passante** (cf. [`SYNTHESE.md`](SYNTHESE.md) §3 :
intensité arithmétique 131–308 FLOP/octet contre un point de bascule à ~563–591). Dans ce régime,
le coût dominant est la lecture du KV cache — laquelle est faite **une seule fois** pour les γ+1
candidats. Le décodage spéculatif est donc **bien plus rentable en contexte long** que le modèle
standard ne le suggère : il amortit le terme dominant, pas un terme secondaire.

## 3. Résultat original : la sparsité casse cet amortissement

**[INFÉRENCE]** L'amortissement suppose que les γ+1 candidats lisent **le même** KV cache. En
attention creuse, chaque position candidate exécute son propre top-k et sélectionne
potentiellement des entrées différentes. La vérification doit lire l'**union** des sélections.
Le coût de vérification n'est plus 1 mais `|union| / k`.

**[FAIT] Mesure sur attention réelle** (GPT-2, top-64, positions de requête voisines) — taille de
l'union rapportée à k, et efficacité d'amortissement résultante :

| γ | dense | creux m=1 (type DSA) | creux m=4 (type CSA) |
|---:|---:|---:|---:|
| 1 | 100 % | 76,8 % | **87,3 %** |
| 2 | 100 % | 65,3 % | **80,5 %** |
| 3 | 100 % | 58,5 % | **75,9 %** |
| 4 | 100 % | 53,5 % | **72,8 %** |
| 6 | 100 % | 46,2 % | **68,3 %** |
| 8 | 100 % | 41,8 % | **65,2 %** |

**[FAIT] Accélération résultante** (modèle borné bande passante, tête de brouillon à coût 0,1) :

| α | γ | dense | creux m=1 | creux m=4 |
|---:|---:|---:|---:|---:|
| 0,7 | 4 | 1,98× | 1,22× | 1,56× |
| 0,7 | 8 | 1,78× | **1,00×** | 1,37× |
| 0,8 | 6 | 2,47× | 1,43× | 1,92× |
| 0,9 | 8 | 3,40× | 1,92× | 2,63× |

## 4. Trois conséquences

**[FAIT] (a) La spéculation peut être entièrement annulée.** À α=0,7 et γ=8 sous sparsité par
token, l'accélération tombe à 1,00× : le coût de lecture de l'union annule exactement le gain.
Un praticien qui active EAGLE-3 sur un modèle à attention creuse fine-grain et mesure « aucun
gain » n'a pas un bug — il a cette interaction.

**[FAIT] (b) La sparsité abaisse le γ optimal.** À α=0,7, l'optimum dense est γ=4 (1,98×) ;
l'optimum creux m=1 est γ=2–3 (1,26×). **Les réglages de γ publiés pour des modèles denses ne se
transposent pas** aux modèles creux et doivent être re-calibrés à la baisse.

**[INFÉRENCE] (c) La granularité par blocs réconcilie partiellement les deux techniques.** À γ=4,
m=4 conserve 72,8 % d'amortissement contre 53,5 % pour m=1, parce que des requêtes voisines
tombent plus souvent sur les *mêmes blocs* que sur les *mêmes tokens* — la compression agit comme
un lissage de la sélection. **C'est un argument supplémentaire en faveur de CSA que DeepSeek ne
formule pas**, alors que V4 emploie simultanément CSA et MTP (multi-token prediction, depth 1,
§4.2.1) : leur choix de granularité par blocs rend leur propre MTP viable.

**[HYPOTHÈSE]** On peut aller plus loin : **forcer explicitement le partage de sélection** entre
les positions candidates — calculer un top-k unique sur la position de départ (ou l'union des
scores) et l'imposer aux γ+1 candidats. On retrouverait un amortissement de 100 % au prix d'une
légère baisse du taux d'acceptation sur les positions éloignées. Le compromis est presque
certainement favorable dès γ≥4. **Non testé** — c'est la piste la plus immédiatement actionnable
de ce rapport (voir Q2).

## 5. Sécurité : Mistletoe, effondrement furtif du spéculatif

**[FAIT]** Mistletoe [2605.14005] (ajouté au corpus) : un adversaire ajoute au prompt un **suffixe discret court** (δ ∈ V^m) qui ne change ni la qualité ni la perplexité des sorties (un filtre à seuil KL rejette les candidats trop dérivants), mais fait s'effondrer la longueur acceptée moyenne τ — le mécanisme central de l'accélération spéculative — donc le speedup, sans corruption visible des sorties.

Chiffres [2605.14005, cibles Vicuna-7B/13B, frameworks EAGLE et Medusa] :
- MT-bench : speedup ÷1,89 (réduction relative 51,7 %) ; HumanEval/GSM8K : speedup ÷2,12 et ÷2,20, τ en baisse de 1,21 et 1,13 ;
- Exemples : EAGLE 5,47×→1,83× (GSM8K) et →2,77× (HumanEval) ; 3,44×→2,38× (Vicuna-7B, MT-bench) ; Medusa Vicuna-13B 3,26×→1,48× (MT-bench), 3,42×→1,42× (HumanEval) ;
- Le seul composant L_rej fait déjà chuter 5,47×→3,30× (τ 5,95→3,41).

**[INFÉRENCE]** Implication système : sur un serving multi-tenant, le spéculatif devient une surface d'attaque par déni de service économique — le coût/token de la victime remonte vers celui du décodage vanilla + draft gaspillé, indétectable par les métriques de qualité usuelles. Contremesures naturelles non étudiées dans le papier : plafonner le budget draft par requête/tenant, surveiller τ par tenant.

## 6. Limites

- L'union est mesurée sur GPT-2 avec un **score oracle**, sur des positions de requête réelles et
  consécutives. Dans un vrai décodage spéculatif, les positions candidates portent des tokens
  *brouillon* (parfois faux), ce qui peut réduire encore le recouvrement. Mes chiffres sont donc
  plutôt **optimistes**.
  **Répliqué sur SmolLM2-135M** : l'amortissement y est encore **plus faible** — 45,2 % (m=1) et
  62,3 % (m=4) à γ=4, contre 53,5 % et 72,8 % sur GPT-2 ; 33,7 % à γ=8 pour m=1. L'interaction
  négative est donc plus sévère sur architecture moderne, et l'écart en faveur de la granularité
  par blocs se maintient (+17 points).
- Le modèle de coût suppose un régime purement borné par la bande passante et ignore le coût de
  l'indexeur pour les positions candidates (qui, lui, s'amortit bien puisqu'il lit les mêmes clés
  d'indexation). Prendre cela en compte **améliorerait** le cas creux ; l'ordre des conclusions est
  robuste, pas les valeurs exactes.
- α est traité comme un paramètre libre et non mesuré ici.


## Addenda v2 — compléments vérifiés

- Date : 2026-08-30 · Corpus : 31 papiers arXiv
- [FAIT] EAGLE-3 [2503.01840] (en production dans SGLang/vLLM/TensorRT-LLM selon INDEX.md) abandonne la contrainte d'EAGLE-1/2 : le draft ne doit plus prédire les features du niveau supérieur ; il consomme des features multi-niveaux (low/mid/high, sélectionnées à travers les couches intermédiaires) — « training-time test ». Scaling law : le speedup croît avec le volume de données d'entraînement du draft (8× plus de données qu'EAGLE-2 → +1.4× sur EAGLE-2 à batch 1).
- - Moyennes EAGLE-3 par cible : Vicuna-13B 5.51× ; Llama-3.1-8B 4.44× ; Llama-3.3-70B 4.12× ; DeepSeek-R1-Distill-Llama-8B 4.16× (tâche maximale : 5.01× sur GSM8K — le draft de ce modèle a été entraîné sur OpenThoughts-114k-math, d'où le pic GSM8K) ; extremum par tâche : 6.47× (V13B, HumanEval) à 3.08× (DSL, CNN/DM) ;
- - Le papier écrit « approximately 3.0x-6.5x » vs décodage autorégressif naïf ; EAGLE-2 en moyenne V13B : 3.28× → gain EAGLE-3 ≈ ×1.7 sur V13B (paper : « about 1.4x improvement over EAGLE-2 ») ;
- - À température=1 : EAGLE-3 3.45-4.65× de moyenne selon le modèle ; le papier ne compare pas aux méthodes non-lossless (Medusa, PLD) dans ce régime ;
- - EAGLE-3 fonctionne avec des cibles quantifiées (W4A16) — point de déploiement majeur.
- - **Table 1** (harnais de recherche, batch 1, par requête isolée) : 3.0-6.5×.
- - **Table 4** (SGLang production, bs=1, H100, Llama-3.1-8B, MT-bench) : 158.34 tokens/s sans spéculatif → 244.10 avec EAGLE-2 (×1.54) → 373.25 avec EAGLE-3 (**×2.36**). Le speedup réel d'un engine à batch 1 est ~2.4×, pas 5.6×.
- - **Table 3** (SGLang v0.4.4, H100, L31-8B, chaîne=3 sans arbre, batch 2→64) : EAGLE-3 1.81× (b=2), 1.82× (4), 1.62× (8), 1.48× (16), 1.39× (24), 1.32× (32), 1.38× (48), 1.34× (56), 1.38× (64) ; EAGLE-1 1.40× (b=2) puis 0.93× (24), 0.94× (32), 0.88× (48), 0.99× (56), 0.99× (64) — **négatif dès b=24**.
- - **Table 5** (vLLM, A100, L31-8B, chaîne=2, batch 2→56) : EAGLE-3 1.75× (b=2) → 1.42× (24) → 1.36× (32) → 1.21× (48) → **1.01× (56)** ; EAGLE 1.30× → 0.93× (32) → 0.82× (48) → 0.71× (56). Le pic d'efficacité d'EAGLE-3 est à b=56 et celui d'EAGLE à b=24 (§4.4 du papier).
- [INFÉRENCE] Trois lectures : (a) le speedup « abstract » (3-6.5×) n'est observable que dans un harnais batch-1 isolé ; le même modèle sur le même dataset dans SGLang donne ×2.36 à bs=1 — l'écart (facteur ~2.3) est le prix des overheads d'un engine réel (scheduling, kernels, gestion mémoire) ; (b) à partir de b≈24-32, le gain tombe sous 1.4× pour EAGLE-3 et sous 1.0× pour EAGLE-1 — la frontière de rentabilité dépend du hardware (A100 : négatif plus tôt) ; (c) la remontée 1.34-1.38× à b=56-64 dans la table SGLang n'est pas expliquée dans le papier (possiblement un artefact de configuration chaîne=3). Conclusion opérationnelle inchangée : le spéculatif est un outil de latence, pas de throughput.
- [FAIT] **PARSE — Parallel Prefix Verification** [2605.04263] : vérifie des préfixes alternatifs en parallèle au lieu de tokens consécutifs — pertinent quand la charge réutilise de longs préfixes (RAG, agents multi-tours). 1.3-4.3× (Qwen3-235B cible, draft 8B) ; MMLU 1.62×, MMLU-Pro 1.48×, GPQA 1.36×.
- [FAIT] **EAGLE-Pangu** [2603.08088] : portage Ascend NPU ; speedup moyen 1.27× (max 1.48× à M=16, D_max=10) ; p90 1.84×, p99 2.46× — l'arbre « sûr » sous-performe sur NPU car le branching n'y est pas amorti comme sur GPU (warp divergence).
- [ATTENTION — source hors corpus] `06_ressources_en_ligne.md` mentionne l'attaque **Mistletoe** [arXiv 2605.14005] contre les déploiements spéculatifs partagés. Le papier n'est PAS dans le corpus : impossible de vérifier le mécanisme exact ici. À ajouter en priorité au corpus (cf. LACUNES_CORPUS.md). [FAIT que la référence figure dans la fiche de ressources ; HYPOTHÈSE pour le mécanisme]
- | EAGLE-1/2 | cités dans 2503.01840 | feature-level | idem EAGLE-3 | ~3.0-4.2× (Table 1) | <1.0× dès b=24 (H100, Table 3) ; 0.71× à b=56 (A100, Table 5) | A100/H100 | référence |
- | EAGLE-3 | 2503.01840 | multi-niveau | V13B, L31-8B, L33-70B, DSL8B | **~3.0-6.5×** (Table 1) ; **2.36× réel** SGLang bs=1 (Table 4 : 373 vs 158 t/s) | 1.32× @b=32, 1.38× @b=64 (Table 3) ; 1.01× @b=56 vLLM A100 (Table 5) | A100/H100 | production vLLM/SGLang/TRT-LLM |
- | PARSE | 2605.04263 | préfixes parallèles | Qwen3-235B (draft 8B) | 1.3-4.3× | — | — | MMLU 1.62×, GPQA 1.36× |
- | MTP (V3/V4) | 2412.19437 / 2606.19348 | module intégré | DeepSeek V3/V4 | intégré | co-trained, compatible batch | H800 | le draft est un sous-module |
- 1. **Spéculatif vs batching** (tension majeure) : 3.0-6.5× dans le harnais (Table 1) mais seulement 2.36× dans SGLang à bs=1 (Table 4 : 373.25 vs 158.34 t/s) ; 1.32× à b=32 ; <1.0× pour EAGLE-1 dès b=24. Il y a DEUX tensions emboîtées : abstract vs engine (facteur ~2.3), et batch 1 vs batch élevé (facteur ~1.4 à 4.4). La question opérationnelle n'est pas « quel spéculatif » mais « spéculatif ou pas, selon la charge ». [FAIT pour les chiffres ; frontières dépendantes du modèle/matériel]
- 1. Frontière batch du spéculatif par modèle/matériel : sweep batch {1..128} × {EAGLE-3, MTP-V4-Flash} sur 8×H20, mesurer throughput/requête et TPOT. Croisement avec la charge réelle (distribution de longueurs).

## 7. Références du corpus

- `04_speculative_decoding/EAGLE-3_2503.01840.pdf`
- `04_speculative_decoding/Parallel-Prefix-Verification_2605.04263.pdf`
- `04_speculative_decoding/Cross-Attention-Speculative-Decoding_2505.24544.pdf`
- `04_speculative_decoding/EAGLE-Pangu_Ascend-NPU_2603.08088.pdf`
- `05_serving_systems/StreamServe_2604.09562.pdf` — flux spéculatifs en serving désagrégé
- Code : [`_tools/exp_C_spec_x_sparse.py`](../_tools/exp_C_spec_x_sparse.py)
- Mistletoe — Stealthy Acceleration-Collapse Attacks on Speculative Decoding ([2605.14005](https://arxiv.org/abs/2605.14005))
