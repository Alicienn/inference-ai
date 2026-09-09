# Optimisation de l'inférence IA — corpus et analyses

**64 papiers arXiv** organisés par thème, **8 analyses originales** et **1 POC de recherche**,
avec code reproductible. Constitué le 30 août 2026.

---

## ▶ Commencer ici

| Document | Contenu |
|---|---|
| **[17_papier/asp.pdf](17_papier/asp.pdf)** | **Le papier.** *Bytes Are Not Milliseconds* — 9 pages, format MLSys/acmart. Selection de blocs en deux passes : 13-25x d'octets economises, 1,2-1,7x plus lent de bout en bout, et l'explication complete. Sources : `asp.tex`, `refs.bib` (35 refs verifiees), `figures.py`. |
| **[07_analyses/SYNTHESE.md](07_analyses/SYNTHESE.md)** | **Le document principal.** État de l'art transversal + 8 résultats originaux, dont la vérification indépendante des chiffres de DeepSeek-V4. |
| **[12_poc/RAPPORT_POC.md](12_poc/RAPPORT_POC.md)** | **POC de recherche.** Résumés de blocs pour l'attention creuse : nouveau cadre (quantification de mesure sous transformée de Laplace), un théorème, un résultat positif robuste (coreset > covariance, 4,5× moins cher que COBS), quatre résultats négatifs et leur cause commune. |
| **[07_journal_recherche.md](07_journal_recherche.md)** | **Journal de recherche** — raisonnements en cours, dérivations, pistes abandonnées et autocorrections. |
| **[12_poc/CONCEPT_ASP.md](12_poc/CONCEPT_ASP.md)** | **Concept nouveau : ASP**, allocation séquentielle de précision. Formalisation, algorithme, gain mesuré, limites, et les 8 propositions classées par ambition et confiance. |
| **[16_gpu/RAPPORT_GATHER.md](16_gpu/RAPPORT_GATHER.md)** | **Mesures GPU réelles** sur 3 architectures mémoire (gfx1152/LPDDR5, T4/GDDR6, A100/HBM2e) : le gather irrégulier d'ASP coûte 5 %, puis 2 %, puis rien. Le seuil de granularité disparaît sur A100 — et deux modèles prédictifs de ce seuil sont réfutés. |
| [07_analyses/QUESTIONS_OUVERTES.md](07_analyses/QUESTIONS_OUVERTES.md) | 10 pistes de recherche classées par impact × faisabilité, avec protocoles |
| [07_analyses/METHODES.md](07_analyses/METHODES.md) | Reproduction, hypothèses explicites, limites, hypothèses infirmées |
| [07_analyses/LACUNES_CORPUS.md](07_analyses/LACUNES_CORPUS.md) | Ce qui manquait, ce qui a été ajouté, ce qui manque encore |
| [06_ressources_en_ligne.md](06_ressources_en_ligne.md) | Documentation, dépôts, blogs pour la veille |

**Notes par axe :** [attention creuse](07_analyses/axe1_attention_creuse.md) ·
[KV cache](07_analyses/axe2_kv_cache.md) ·
[décodage spéculatif](07_analyses/axe3_decodage_speculatif.md) ·
[serving](07_analyses/axe4_serving.md)

---

## ▶ Les huit résultats en une page

1. **[FAIT]** À 1M de contexte, **90 % des FLOPs de décodage de DeepSeek-V3.2 sont consommés par
   son propre indexeur** — pas par l'attention. L'attention creuse déplace le goulot d'étranglement
   vers le mécanisme de sélection.
2. **[FAIT]** Les chiffres de DeepSeek-V4 sont **reproductibles** à partir des seuls
   hyperparamètres publiés : paramètres à ±1,7 %, ratios de KV cache à ±4 %, FLOPs à ±10 %.
3. **[FAIT]** Il faut distinguer **KV résident** (capacité) et **KV lu par token** (latence). La
   seconde métrique, absente de la littérature, donne à V4-Flash un avantage de **18,6×** au lieu
   des 13,7× annoncés.
4. **[INFÉRENCE]** La compression par blocs ne marche **pas** par regroupement spatial (ρ=0,21 au
   décalage 1, 30 % d'aiguilles isolées) mais par préservation de la détectabilité. Le coût est une
   dilution de budget, bornée — d'où l'arbitrage favorable.
5. **[INFÉRENCE]** Faiblesse cachée de CSA : le **trou de Jensen** (jusqu'à −21,8 %). L'entropie du
   compresseur, non discutée dans le papier, est l'hyperparamètre critique. Optimum à β≈0,5, ni
   moyenne ni max.
6. **[FAIT]** En quantification du KV, **l'allocation de bits vaut ~2 bits gratuits** (−73 à −82 %
   d'erreur). L'énergie par bloc RoPE varie de **793×**, avec un gradient de 35× entre basses et
   hautes fréquences.
7. **[FAIT, original]** **Décodage spéculatif et attention creuse se nuisent** : l'amortissement
   tombe à 41,8 % et l'accélération peut s'annuler (1,00×). La granularité par blocs les réconcilie
   partiellement (65,2 %).
8. **[FAIT]** L'intuition « le transfert KV interdit la désagrégation » est **fausse** à prefill
   froid (1,9 % du coût) et **vraie** sous réutilisation de préfixe (937 ms contre 66 ms).

> **Validation croisée.** Les résultats 4, 5 et 7 sont répliqués sur une seconde architecture
> (SmolLM2-135M : RoPE, GQA) en plus de GPT-2. Ils tiennent tous, dans le même sens et souvent
> plus nettement — le regroupement spatial y est encore plus faible (ρ=0,129) et l'interaction
> spéculation×sparsité encore plus sévère (33,7 % d'amortissement à γ=8). Les **amplitudes**
> varient d'un modèle à l'autre, pas les **classements**. Voir [SYNTHESE §9](07_analyses/SYNTHESE.md).

---

## ▶ Le corpus

### 01_deepseek — lignée DeepSeek (6)

| Fichier | arXiv | Résumé |
|---|---|---|
| DeepSeek-V4_Million-Token-Context | [2606.19348](https://arxiv.org/abs/2606.19348) | **Papier central.** V4-Flash (284B/13B actifs) et V4-Pro (1,6T/49B), contexte 1M. Attention hybride CSA+HCA, mHC, experts routés en FP4. |
| FlashMemory — Lookahead Sparse Attention | [2606.09079](https://arxiv.org/abs/2606.09079) | LSA : *Neural Memory Indexer* prédictif, ne garde en HBM que les chunks critiques. 2,8× débit à 1M. |
| DeepSeek-V3.2 (DSA) | [2512.02556](https://arxiv.org/abs/2512.02556) | DeepSeek Sparse Attention, top-k=2048, O(L²)→O(Lk) |
| mHC | [2512.24880](https://arxiv.org/abs/2512.24880) | Hyper-connexions contraintes sur le polytope de Birkhoff |
| DeepSeek-V3 Technical Report | [2412.19437](https://arxiv.org/abs/2412.19437) | MLA, DeepSeekMoE, FP8, MTP — la fondation |
| NSA — Native Sparse Attention | [2502.11089](https://arxiv.org/abs/2502.11089) | Sparsité entraînable alignée matériel, précurseur de DSA |

### 02_surveys — états de l'art (6)

`Survey_LLM-Inference-Systems` (2506.21901) · `Survey_Inference-Engines` (2505.01658) ·
`Taming-the-Titans` (2504.19720) · `Survey_Model-Routing-Cascading` (2603.04445) ·
`Review_KV-Cache-Compression` (2508.06297) · `Reasoning-Model-Serving-Empirical` (2510.18672)

### 03_attention_kv_cache — attention efficace et compression KV (10)

`TurboQuant` (2504.19874) · `RoPE-Aware-Bit-Allocation` (2606.24033) · `RDKV` (2605.08317) ·
`Quantization-vs-Rank-Reduction` (2604.11501) · `ARKV` (2603.08727) · `SpargeAttention2` (2602.13515) ·
`DashAttention` (2605.18753) · `FlashAttention-3` (2407.08608) · `StreamingLLM` (2309.17453) ·
`Infini-attention` (2404.07143)

### 04_speculative_decoding — décodage spéculatif (5)

`EAGLE-3` (2503.01840) · `Parallel-Prefix-Verification` (2605.04263) ·
`Cross-Attention-Speculative-Decoding` (2505.24544) · `EAGLE-Pangu` (2603.08088) ·
`Mistletoe` (2605.14005 — attaques sur la spéculation)

### 05_serving_systems — serving et désagrégation (11)

`vLLM/PagedAttention` (2309.06180) · `SGLang/RadixAttention` (2312.07104) · `DistServe` (2401.09670) ·
`Mooncake` (2407.00079) · `Sarathi-Serve` (2403.02310) · `FlashInfer` (2501.01005) ·
`DuetServe` (2511.04791) · `StreamServe` (2604.09562) · `PD-Aggregation-or-Disaggregation` (2508.01989) ·
`FlowPrefill` (2602.16603) · `Load-Aware-Prefill-Deflection` (2607.02043)

### 08_quantization_poids — quantification des poids (4)

`AWQ` (2306.00978) · `GPTQ` (2210.17323) · `QuaRot` (2404.00456) · `SpinQuant` (2405.16406)

### 09_moe_inference — inférence MoE (3)

`Mixtral-Offloading` (2312.17238) · `Towards-MoE-Deployment` (2303.06182) ·
`Inter-Layer-Expert-Affinity` (2401.08383)

### 10_benchmarks_eval — évaluation (2)

`RULER` (2404.06654) · `Energy-to-Token-Evaluation` (2605.11733)

### 13_selection_blocs — sélection de blocs, état de l'art (6)

`COBS` (2607.09052 — expansion en cumulants, l'antériorité directe du POC) ·
`Uncertainty-Gated-Block-Selection` (2607.07724) · `XAttention` (2503.16428) ·
`BA-Att` (2605.19726) · `SALE` (2505.24179) · `RRAttention` (2602.05853)

### 14_hierarchique — sélection hiérarchique et multi-résolution (6)

`AsyncTLS` (2604.07815) · `HiSparse` (2608.07009) · `HieraSparse` (2604.16864) ·
`SALS` (2510.24273) · `SeerAttention-R` (2506.08889) · `Sparse-Attn-Multi-Context-KV` (2508.11661)

### 15_theorie_selection — théorie de la sélection sous bruit (3, hors domaine)

`Bubeck-Munos-Stoltz` (0802.2655 — regret simple vs identification) ·
`Carpentier-Locatelli` (1605.09004 — borne inférieure en budget fixe) ·
`Jamieson lil'UCB` (1312.7308)

### 11_materiel — pistes matérielles (2)

`3D-DRAM-LLM-Accelerators` (2604.08044) · `SMEPilot` (2606.16332)

---

## ▶ Code et données

| Chemin | Contenu |
|---|---|
| `_tools/` | 10 scripts (modèle analytique de coût, roofline, 5 expériences, validation croisée) |
| `12_poc/code/` | 11 scripts du POC (capture Q/K réels, estimateurs de blocs, diagnostics, ablations) |
| `12_poc/resultats/` | Q/K capturés, résultats JSON, sorties brutes |
| `16_gpu/` | Bancs GPU (gfx1152, T4, A100, H200), source unique CUDA/HIP, noyau ASP fusionné (`fused.cu`), intégration Qwen3-8B (`e2e_qwen.py`) |
| `17_papier/` | Papier LaTeX complet : `asp.tex`, `refs.bib`, `figures.py`, `figs/` (5 figures), `asp.pdf` |
| `18_refs_papier/` | Les 9 références du papier dont le titre a été vérifié par extraction de la 1re page |
| `_txt/` | Texte extrait des PDFs, interrogeable via `_tools/corpus_search.py` |
| `07_analyses/_sorties_brutes/` | Sorties console horodatées de chaque script |
| `07_analyses/*.json` | Résultats structurés des expériences |

Exemple :

```bash
cd C:\Users\alici\Downloads\inference_opti\_tools && python model_inference_cost.py
```

**Environnement :** Python 3.11, numpy, torch CPU, transformers. **Aucun GPU requis.**
Les expériences empiriques utilisent GPT-2 et SmolLM2-135M sur wikitext.

## Livrables & état (v2, fin de session)
- Livrables finaux aux noms demandés par la mission, à la racine du dossier ; versions canoniques dans `07_analyses/` (avec `METHODES.md`).
- `_archive_v1/` : notes v1 remplacées (erreurs corrigées en v2 : recette 943,7B réattribuée à V3.2 ; « 70 Ko/token » re-tagué [INFÉRENCE] ; « PD-Pai » → TaiChi ; plage EAGLE-3 corrigée).
- Texte extrait : `_txt/` et `_extracted/` ; scripts : `_tools/`.


## Axe 20 — Erreur de sortie d'attention (session 12)

`20_sortie_attention/RAPPORT_SORTIE.md` : decomposition de l'erreur de sortie d'attention
sous quantification et sous selection (Qwen3-8B, SmolLM2-135M, masque causal, fenetre
locale exacte). Resultat : la quantification 4 bits coute 2,6-6,8x moins que la selection ;
4 -> 8 bits n'ameliore le pipeline que de <= 12 % ; la cle moyenne echoue sur les tetes de
recuperation (Jensen gap). Code : `20_sortie_attention/code/err_attention2.py`,
`qwen_run.py`. Donnees : `20_sortie_attention/resultats/err_attention_{smol,qwen}_v2.json`.
Verification : cellule 9 du runbook 9944d27c (runbook_run_8ea7e88b).
