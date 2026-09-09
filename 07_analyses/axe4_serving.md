# Axe 4 — Serving : désagrégation, scheduling, et économie du déploiement

## 1. Le paysage

| Système | Idée structurante | Papier |
|---|---|---|
| **vLLM / PagedAttention** | KV cache paginé, pas de fragmentation | [2309.06180](https://arxiv.org/abs/2309.06180) |
| **DistServe** | Séparer prefill et decode sur du matériel dédié | [2401.09670](https://arxiv.org/abs/2401.09670) |
| **Sarathi-Serve** | Chunked prefill, batching sans stalls (2,6–5,6× sur le compromis débit/latence) | [2403.02310](https://arxiv.org/abs/2403.02310) |
| **Mooncake** | Architecture *KVCache-centric*, routage conscient des préfixes | [2407.00079](https://arxiv.org/abs/2407.00079) |
| **FlashInfer** | Moteur de kernels d'attention personnalisables | [2501.01005](https://arxiv.org/abs/2501.01005) |
| **DuetServe / StreamServe / FlowPrefill** | Raffinements 2026 du compromis P/D | 2511.04791, 2604.09562, 2602.16603 |

**[FAIT]** Le corpus contient aussi un papier qui *unifie* les deux régimes
([2508.01989](https://arxiv.org/abs/2508.01989)) : la question n'est plus « agréger ou
désagréger » mais « selon quel régime de charge basculer ».

## 2. Le décodage est borné par la bande passante — toujours

**[FAIT]** Mesure roofline (voir [`_sorties_brutes/02_roofline_decode.txt`](_sorties_brutes/02_roofline_decode.txt)) :
intensité arithmétique de l'attention en décodage = 131 (V3.2) à 308 (V4-Pro) FLOP/octet, contre
un point de bascule de **591 FLOP/octet sur H800** et **563 sur B200**. Aucun de ces modèles
n'approche le régime compute-bound en décodage.

**[INFÉRENCE]** Deux conséquences pour le praticien :
1. Optimiser les FLOPs d'attention en décodage **est sans effet**. Seule compte la réduction des
   octets lus. C'est ce qui rend la métrique « KV lu par token » (axe 2, §2) opérationnelle.
2. Le point de bascule **se dégrade** sur matériel récent : le B200 multiplie le calcul par 2,3
   et la bande passante par 2,4, donc le ratio reste quasi constant — mais les architectures, elles,
   augmentent leur intensité (131 → 308 de V3.2 à V4-Pro). **La sparsité rapproche du régime
   compute-bound**, ce qui est précisément ce qu'on veut : elle convertit un problème de bande
   passante en problème de calcul, sur lequel le matériel progresse plus vite.

## 3. Le transfert KV en désagrégation : une intuition fausse, un régime vrai

**[HYPOTHÈSE initiale, INFIRMÉE]** J'ai supposé que la taille du KV cache rendait la désagrégation
inter-nœud impraticable en contexte long.

**[FAIT]** Faux à **prefill froid**. Part du transfert dans le temps total (InfiniBand 400G) :

| contexte | V3.2 | V4-Flash |
|---:|---:|---:|
| 8 000 | 9,9 % | 0,7 % |
| 128 000 | 6,4 % | 0,6 % |
| 1 000 000 | **1,9 %** | **0,3 %** |

Le prefill à 1M coûte ~49 s (V3.2, 16×H800, 40 % de MFU) contre 0,94 s de transfert. Le calcul
domine, et la part du transfert **décroît** avec le contexte puisque le prefill croît plus vite
que le cache. La désagrégation est donc largement viable — ce que confirme son adoption.

**[FAIT] Le régime où le transfert devient décisif est la réutilisation de préfixe.** C'est la
thèse même de Mooncake : le gain vient des *hits* de cache (multi-tours, agents, RAG). En cas de
hit, il n'y a plus de prefill à amortir — le transfert est **100 %** du coût.

| Servir un préfixe de 1M déjà en cache | NVLink | IB 400G | Eth 100G |
|---|---:|---:|---:|
| DeepSeek-V3.2 | 52 ms | **937 ms** | 3 748 ms |
| DeepSeek-V4-Flash | 4 ms | **66 ms** | 264 ms |
| DeepSeek-V4-Pro | 5 ms | 96 ms | 383 ms |

**[INFÉRENCE]** Avec V3.2, un hit de cache à 1M coûte ~0,9 s de transfert inter-nœud — assez pour
annuler l'intérêt du cache et forcer à confiner la réutilisation au domaine NVLink (donc à un seul
nœud, ce qui limite drastiquement le taux de hit atteignable). Avec V4-Flash (66 ms), la
réutilisation de préfixe redevient rentable **sur réseau standard**, donc à l'échelle d'un cluster.

**C'est le point de jonction des deux littératures :** un choix d'architecture (compresser le KV)
débloque une technique de serving (le cache de préfixes distribué). Ni le papier V4 ni Mooncake ne
font ce lien — ils ne se citent pas.

**[FAIT]** V4 [2606.19348 §3.5.2] définit trois stratégies de placement du KV en désagrégation P/D — le nœud de prefill cache le KV complet (full), des checkpoints intermédiaires (checkpoint), ou rien (zero) — et note que le KV SWA est ~8× plus volumineux que le KV CSA/HCA compressé : le transfert KV est le goulot de la désagrégation longue-contexte. LSA [2606.09079 §4] y répond par un transfert sélectif des seuls blocs retenus (×2,7 de concurrence en P/D).

## 4. Seuils de conception

**[FAIT]** Taille maximale de KV cache pour que le transfert reste sous 10 % du temps de prefill
(calculé sur V4-Flash, 4×H800, 40 % MFU, contexte 1M) :

| lien | KV max admissible |
|---|---:|
| NVLink 4 (intra-nœud) | 2 377 GB |
| InfiniBand NDR 400G | 132 GB |
| Ethernet 200G | 66 GB |
| Ethernet 100G | 33 GB |

**[INFÉRENCE]** Ces seuils sont confortables à prefill froid — d'où le §3. Ils deviennent
contraignants dès qu'on vise un taux de hit élevé sur le cache de préfixes, puisqu'il faut alors
comparer le transfert non plus au prefill mais au **TTFT cible** (typiquement 200–500 ms). À ce
critère, seul V4-Flash passe sur IB.

**[FAIT]** TaiChi [2508.01989] (Huawei Cloud + CUHK) tranche le débat : **aucun des deux régimes n'est universellement optimal**. SLO TTFT-strict → l'agrégation gagne ; SLO TPOT-strict → la désagrégation gagne ; SLO équilibrés → ni l'un ni l'autre (il faut borner l'interférence : <273,77 tokens préfill par token de sortie pour rester optimal). TaiChi unifie les deux en basculant par phase : goodput +77 % vs SOTA, TTFT −13,2×, TPOT −1,69× ; +9-47 % vs agrégation pure et +29-77 % vs désagrégation pure (QPS élevé). C'est la formalisation opérable du « régime » défendu en §4.

## 5. Empreinte et capacité

**[FAIT]** Poids en mémoire, experts routés en FP4 pour la série V4 (§1 du papier V4 : « les
paramètres des experts routés utilisent la précision FP4 ») :

| modèle | poids | GPU minimum (B200 192 GB) |
|---|---:|---:|
| DeepSeek-V3.2 (FP8) | 625 GB | 4 |
| **DeepSeek-V4-Flash (FP4)** | **135 GB** | **1** |
| DeepSeek-V4-Pro (FP4) | 770 GB | 8 |

**[INFÉRENCE]** V4-Flash est un modèle de 284 B paramètres qui **tient sur un seul B200**. C'est
un changement de nature pour le déploiement : plus de parallélisme de tenseurs, plus de
communication inter-GPU sur le chemin critique, et une granularité de scaling à 1 GPU. Le couple
« MoE très creux (13 B actifs sur 284 B) + poids en FP4 » est ce qui rend cela possible — c'est
sans doute l'aspect le plus sous-estimé de la conception de V4-Flash.

**[HYPOTHÈSE]** Le vrai concurrent de V4-Flash n'est pas V3.2 mais un modèle dense de ~13–30 B :
même coût de calcul par token, mais V4-Flash apporte la capacité d'un 284 B. Le prix à payer est
la mémoire (135 GB contre ~30 GB). L'arbitrage bascule dès qu'on sert **beaucoup** de requêtes
concurrentes, puisque les poids s'amortissent sur le batch tandis que le calcul, lui, ne s'amortit
pas. À faible concurrence, un modèle dense reste préférable. **Non vérifié.**

## 6. Limites

- MFU de 40 % supposée uniformément ; les valeurs réelles varient beaucoup selon le noyau et la
  forme du batch.
- Le modèle de temps de prefill intègre grossièrement le coût croissant de l'attention sur 25
  points ; il ignore le chunked prefill et le recouvrement calcul/communication.
- Les temps de transfert supposent une bande passante nominale sans surcoût de protocole ni
  contention — donc **optimistes** d'un facteur 1,3 à 2 en pratique, ce qui **renforce** la
  conclusion du §3 sur la réutilisation de préfixe.
- Aucune mesure réelle : la machine ne dispose d'aucun GPU. Tout est analytique.


## Addenda v2 — compléments vérifiés

- Date : 2026-08-30 · Corpus : 31 papiers arXiv
- [FAIT] **vLLM/PagedAttention** [2309.06180] : pagination du KV cache (blocs logiques de 16 tokens, table de pages) — élimine la fragmentation mémoire externe/interne (>30% gaspillage avant). 2-4× throughput vs baselines ; jusqu'à ×l'utilisation mémoire. C'est le socle de tout le reste.
- [FAIT] **Sarathi-Serve** [2403.02310] : chunked prefill + stall-free batching — piggybacking des decodes sur les chunks de prefill. 2.6-5.6× capacité de serving (Mistral-7B 2.6×, 180B 5.6×). Réduit l'interférence sans séparer les ressources.
- [FAIT] **DistServe** [2401.09670] : désagrégation stricte — parce que TTFT et TPOT ont des SLO distincts (le prefill détermine le TTFT, le decode le TPOT), séparer physiquement permet d'optimiser chaque phase indépendamment. Jusqu'à 7.4× plus de requêtes/s ou 12.6× de SLO plus strict vs vLLM/Orca chunked.
- [FAIT] **Mooncake** [2407.00079] : KVCache-centric disaggregated serving — le cluster est organisé autour du cache (CPU DRAM ~1 To × 20k blocs, NVMe multi-PB, RDMA) ; réutilisation de préfixe multi-tenant (agents, multi-tours). +525% throughput (Kimi) ; hit ratio 30%→50% avec 1000→50000 blocs.
- [FAIT] **TaiChi** [2508.01989] : la question n'est pas « désagréger ou pas » mais **quand** : TTFT serré/TPOT relâché → agrégation ; TPOT serré/TTFT relâché → désagrégation ; SLO équilibrés → ni l'un ni l'autre (point-slop). Cadre unifié qui switch dynamiquement : +77% goodput vs SOTA, TTFT -13.2×, TPOT -1.69×. C'est le résultat le plus important de l'axe pour la prise de décision.
- [FAIT] **DuetServe** [2511.04791] : multiplexage **spatial** GPU (partition SM via libsmctrl/MPS) plutôt que temporel — allouer les SM aux prefills/decodes selon la pression SLO ; +1.3× throughput total.
- [FAIT] **FlashInfer** [2501.01005] (MLSys 2025 best paper) : moteur de kernels d'attention personnalisables (API Load/Store/Plan/Run, BlockSparse) ; 29-69% ITL réduit vs Triton, 28-30% latence long-context, 13-17% speedup serving ; +15.95% throughput. Couche logicielle qui rend compatibles kernels creux/quantisés/hétérogènes.
- [FAIT] V4 [2606.19348, §3.5.2] consacre une section entière au KV transfer en désagrégation on-disk : le SWA « ~8× plus volumineux » que le CSA/HCA compressé pose un problème de bande passante — d'où 3 stratégies (full caching / checkpoint périodique / zero + recompute). LSA [2606.09079, §4] montre que la prédiction lookahead rend le transfert asynchrone et sélectif : 2.7× concurrency en PD-disaggregated serving sur 8×H20. Mooncake traite le transfert par RDMA + cache hiérarchique. [INFÉRENCE] Le KV transfer est LE coût caché de la désagrégation : il convertit un gain compute en problème de bande passante réseau — c'est pourquoi TaiChi (switch dynamique) et DuetServe (multiplexage SM) existent.
- | vLLM/PagedAttention | 2309.06180 | 2023 | Pagination KV (16-token blocks) | 2-4× throughput | vs FasterTransformer/Orca | fragmentation mémoire élevée | A100 |
- | Sarathi-Serve | 2403.02310 | 2024 | Chunked prefill + piggyback decode | 2.6-5.6× capacité | vs vLLM (charge 2.6-5.6×) | interference prefill/decode | A100/H100 |
- | DistServe | 2401.09670 | 2024 | Désagrégation stricte P/D | 7.4× req/s ou 12.6× SLO plus strict | vs vLLM/Orca | SLO TTFT≠TPOT ; coût réseau | A100 |
- | Mooncake | 2407.00079 | 2024 | KVCache-centric + préfixe | +525% throughput | Kimi (75% requêtes en plus) | forte réutilisation de préfixe | A800 cluster + RDMA |
- | TaiChi | 2508.01989 | 2025 | Switch dynamique agg/désagg | +77% goodput, TTFT -13.2×, TPOT -1.69× | vs SOTA (DistServe etc.) | mix de charges ; SLO hétérogènes | A100/H100 |
- | DuetServe | 2511.04791 | 2025 | Multiplexage SM spatial | +1.3× throughput | vs baselines MPS | besoin de partitions SM | H100 (libsmctrl) |
- | StreamServe | 2604.09562 | 2026 | Spéculatif adaptatif par confiance | latence/variance réduites | vs vLLM+SGLang baselines | charge mixte courte/longue | H100 |
- | FlowPrefill | 2602.16603 | 2026 | Préemption operator-level + SJF | réduit HoL blocking des longs prefills | — | longs prefills fréquents | H100 |
- | FlashInfer | 2501.01005 | 2025 | Kernels attention personnalisables | 29-69% ITL, +15.95% throughput | vs Triton | hétérogénéité de charge | H100/H200 |
- 1. **Agrégation vs désagrégation** : DistServe [2401.09670] argumente fortement pour la désagrégation (7.4×) ; TaiChi [2508.01989] montre que c'est conditionnel — agrégation gagne si TTFT serré, désagrégation si TPOT serré, aucun des deux aux points équilibrés. Les deux papiers ne se contredisent pas factuellement (DistServe compare contre un baseline non-chunked), mais la lecture superficielle de DistServe (« désagréger, c'est toujours mieux ») est réfutée par TaiChi. [FAIT pour l'exposition ; pas de contradiction chiffrée directe]
- 3. **Le transfert KV est-il rentable ?** : DistServe/Mooncake supposent un réseau rapide (RDMA) ; LSA montre que le transfert sélectif (indexer lookahead) multiplie la concurrence (2.7×). Mais à 1M tokens, transférer tout le KV, même compressé (V4 : 7% de V3.2), reste ~70 GB/requête — la bande passante inter-nœuds devient le facteur limitant. [INFÉRENCE, à chiffrer]
- 1. **Transfert KV à 1M tokens** : la désagrégation tient-elle quand le KV par requête atteint ~70 GB (V4-Flash) ou ~7 GB (V4 compressé) ? Bande passante nécessaire vs RDMA disponible. Protocole : mesurer goodput en fonction de la bande passante inter-nœuds × longueur de contexte.
- 4. **Le cache de préfixe (Mooncake) survit-il à CSA/HCA ?** Si le modèle ne charge que k=512 blocs, la « réutilisation » de préfixe devient une réutilisation d'indexer — le cache doit indexer les indexers. Question de co-design.

## 7. Références du corpus

- `05_serving_systems/` — l'ensemble des 9 papiers
- `02_surveys/Survey_LLM-Inference-Systems_2506.21901.pdf` — scheduling, batching, désagrégation
- `02_surveys/Survey_Model-Routing-Cascading_2603.04445.pdf` — réduction de coût par routage
- Code : [`_tools/exp_D_serving_economics.py`](../_tools/exp_D_serving_economics.py), [`_tools/roofline_decode.py`](../_tools/roofline_decode.py)
- Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving — TaiChi ([2508.01989](https://arxiv.org/abs/2508.01989))
