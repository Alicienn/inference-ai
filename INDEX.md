# Optimisation de l'inférence IA — corpus et analyses

**64 papiers arXiv** organisés par thème, **8 analyses originales** et **1 POC de recherche**,
avec code reproductible. Constitué le 30 août 2026, dernière réorganisation le 10 septembre 2026.

> **Statut du projet (10 sept. 2026) : la piste ASP est close.** Trois sessions GPU réelles
> (H200 puis RTX 4090 ×2) ont confirmé et caractérisé le résultat négatif du papier — ASP
> économise des octets mais ne devient pas plus rapide, même après correction de deux défauts
> identifiés (voir [Axe 21](21_rtx4090_reel/RAPPORT_RTX4090.md) §8-9). La recherche s'oriente
> maintenant vers **d'autres techniques d'optimisation de l'inférence** (voir tout en bas).

## ▶ Table des matières

- [Commencer ici](#-commencer-ici) — les documents à lire en premier
- [Les huit résultats en une page](#-les-huit-résultats-en-une-page)
- [Le corpus](#-le-corpus) — 64 papiers classés par thème
- [Organisation des fichiers](#-organisation-des-fichiers) — arborescence et archives
- [Code et données](#-code-et-données)
- [Axe 20 — Erreur de sortie d'attention](#axe-20--erreur-de-sortie-dattention-session-12)
- [Axe 21 — Mesures réelles GPU (RTX 4090)](#axe-21--mesures-réelles-gpu-loué-rtx-4090-qwen3-8b--session-13)
- [Journal détaillé (recherche fine, chronologique)](#-journal-détaillé--recherche-fine-sur-le-sélecteur-ordre-chronologique)
- [Prochaine étape : nouvelles techniques](#-prochaine-étape--au-delà-dasp)

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

## ▶ Organisation des fichiers

Arborescence par numéro d'axe (01-15, 18 = corpus PDF ; 16-21 = code et résultats originaux) :

```
01-15, 18_*/     corpus arXiv par thème (PDFs, non versionnés sur GitHub — voir 06_ressources_en_ligne.md)
16_gpu/          bancs GPU réels (gather, e2e, ventilation, correctifs ASP v2), .sh = helpers de connexion vast.ai
17_papier/       papier LaTeX "Bytes Are Not Milliseconds" (asp.tex/pdf, refs.bib, figures.py)
19_loi_octets/   loi octets-précision (quantification, coreset, COBS) sur activations réelles
20_sortie_attention/  recherche fine sur le sélecteur (mean vs max, granularité, sink) — RAPPORT_v28.md = dernière version
21_rtx4090_reel/ synthèse des 3 sessions GPU louées (RTX 4090), tests de correctifs ASP
_tools/          scripts d'analyse transversaux (coût modèle, roofline, validation croisée)
_txt/, _extracted/  texte extrait des PDFs du corpus
_archive_v1/     notes de la 1re passe de recherche, remplacées en v2 (erreurs corrigées, gardées pour traçabilité)
_archive_scratch/  scripts et logs jetables déplacés hors de la racine le 10 sept. 2026 (voir ci-dessous)
```

**Nettoyage du 10 sept. 2026** : 89 fichiers en vrac à la racine et dans 17_papier/ /
20_sortie_attention/code/ (scripts de patch à usage unique, logs de compilation LaTeX
répétés, anciennes versions de rapport) ont été déplacés — pas supprimés — dans :
- `_archive_scratch/session_patches_racine/` — scripts `_fixN.py`/`_write_*.py` (patchs
  ponctuels déjà appliqués à `asp.tex`/aux rapports)
- `_archive_scratch/latex_logs_17_papier/` — logs de compilation LaTeX répétés (`asp.log`
  reste à sa place, c'est le seul qui compte)
- `20_sortie_attention/_archive_versions/` — RAPPORT_v21 à v27.md (superseded par v28)
- `20_sortie_attention/code/_archive_logs/` — logs bruts `bg_*.err`/`bg_*.log` de jobs de fond

Rien n'a été perdu, tout reste dans le dépôt git et localement.

---

## ▶ Code et données

| Chemin | Contenu |
|---|---|
| `_tools/` | 10 scripts (modèle analytique de coût, roofline, 5 expériences, validation croisée) |
| `12_poc/code/` | 11 scripts du POC (capture Q/K réels, estimateurs de blocs, diagnostics, ablations) |
| `12_poc/resultats/` | Q/K capturés, résultats JSON, sorties brutes |
| `16_gpu/` | Bancs GPU (gfx1152, T4, A100, H200, RTX 4090 ×2), source unique CUDA/HIP, noyau ASP fusionné (`fused.cu`), intégration Qwen3-8B (`e2e_qwen.py`, `e2e_qwen_v2.py` = correctif, `e2e_multistep_final.py` = test décisif), diagnostics (`probe_*.py`) |
| `17_papier/` | Papier LaTeX complet : `asp.tex`, `refs.bib`, `figures.py`, `figs/` (5 figures), `asp.pdf` |
| `18_refs_papier/` | Les 9 références du papier dont le titre a été vérifié par extraction de la 1re page |
| `19_loi_octets/` | Loi octets-précision, frontière de Pareto à deux passes, mise à l'échelle en L |
| `20_sortie_attention/` | Recherche sur le sélecteur (mean vs max, granularité, sink) ; `code/` (~90 scripts), `resultats/` |
| `21_rtx4090_reel/` | Synthèse GPU louée réelle : `RAPPORT_RTX4090.md` (9 sections), tests A-G |
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


## Axe 21 — Mesures réelles GPU loué (RTX 4090, Qwen3-8B) — session 13

`21_rtx4090_reel/RAPPORT_RTX4090.md` : trois sessions GPU louées (< 10 $ au total).
Confirme sur 2e GPU independant (apres H200) que ASP economise 13-23x d'octets mais reste
plus lent en bout en bout (Qwen3-8B reel) ; confirme coreset > COBS sur le vrai 8B ; releve
un artefact de cache L2 dans le micro-benchmark gather isole de `16_gpu/RAPPORT_GATHER.md`.
**Session 3 (finale)** : implémentation et test d'un correctif complet (tampons
incrémentaux + score max, `16_gpu/e2e_qwen_v2.py`) — fonctionne isolément (jusqu'à 1,68×
sur la construction des résumés seule) mais **régresse une fois intégré** dans un vrai
décodage multi-pas (0,59-0,63× contre 0,75-0,85× pour le bug d'origine) : le score max
coûte plus cher par appel que la moyenne qu'il remplace. **Piste ASP fermée** à l'issue de
cette session — voir §8-9 du rapport pour le détail et §[Prochaine étape](#-prochaine-étape--au-delà-dasp).

---

## ▶ Journal détaillé — recherche fine sur le sélecteur (ordre chronologique)

*À partir d'ici, contenu ajouté au fil de l'eau par la recherche sur le sélecteur ASP
(mean vs max, granularité de bloc, structure sink+fond). Chaque section est un point de
mesure daté, pas une entrée d'index classique — voir plutôt `20_sortie_attention/RAPPORT_v28.md`
pour la synthèse consolidée de tout ce journal, et [Axe 21](#axe-21--mesures-réelles-gpu-loué-rtx-4090-qwen3-8b--session-13)
ci-dessus pour ce qui a été vérifié sur GPU réel.*

## Axe 20 — Erreur de sortie d'attention (session 12)

`20_sortie_attention/RAPPORT_SORTIE.md` : decomposition de l'erreur de sortie d'attention
sous quantification et sous selection (Qwen3-8B, SmolLM2-135M, masque causal, fenetre
locale exacte). Resultat : la quantification 4 bits coute 2,6-6,8x moins que la selection ;
4 -> 8 bits n'ameliore le pipeline que de <= 12 % ; la cle moyenne echoue sur les tetes de
recuperation (Jensen gap). Code : `20_sortie_attention/code/err_attention2.py`,
`qwen_run.py`. Donnees : `20_sortie_attention/resultats/err_attention_{smol,qwen}_v2.json`.
Verification : cellule 9 du runbook 9944d27c (runbook_run_8ea7e88b).

## RTX 4090 (2026-09-09) — 4e architecture

- [`16_gpu/resultats/rtx4090_raw.txt`](16_gpu/resultats/rtx4090_raw.txt) — sortie brute du banc de gather CUDA (RTX 4090, GDDR6X, pic 933,2 GB/s).
- [`16_gpu/resultats/e2e_qwen_rtx4090.json`](16_gpu/resultats/e2e_qwen_rtx4090.json) — latence de decodage Qwen3-8B, dense contre ASP (16k a 131k).
- [`16_gpu/RAPPORT_GATHER.md`](16_gpu/RAPPORT_GATHER.md) — section 6quater : pas de seuil de gather ; le chiffre d'efficacite mesurait la L2 ; bilan sur quatre GPU.
- [`17_papier/asp.tex`](17_papier/asp.tex) — papier mis a jour (4 architectures, 2 backends, paragraphe sur le confondant L2, croisement 146k).
- Rapport : « La RTX 4090 : le banc mesurait le cache, et un GPU plus lent atteint la parite plus tot » (report_f3aaca65-4d17-498e-a130-49aacf558f34).
- Runbook de reanalyse : `runbook_3bdcabe2-dbc9-47b7-8df3-0f78934846f8`.

## Balayage d'agregations (2026-09-09) — correction de la metrique

- [`20_sortie_attention/code/gpt2_lse.py`](20_sortie_attention/code/gpt2_lse.py) — banc d'induction GPT-2 avec une troisieme agregation (log-sum-exp sur les cles, temperature reglee).
- [`20_sortie_attention/resultats/gpt2_lse.txt`](20_sortie_attention/resultats/gpt2_lse.txt) — 18 configurations (3 budgets x 6 agregations) : le max gagne aux trois budgets ; la LSE est refutee ; la masse retenue S n'est qu'un proxy (lse lambda=2 retient plus de masse et perd plus a 25 %).
- Rapport de correction : « Correction : le maximum reste la meilleure agregation, et la masse retenue n'est qu'un proxy » (report_57712c53-1ec8-43ef-977b-9afb2cf3a627). Corrige la section correspondante du rapport v28.
- [`17_papier/asp.tex`](17_papier/asp.tex) — paragraphe du banc de recuperation corrige (15 pages, 742 910 octets).

## Score sensible a la valeur (2026-09-09) — refutation

- [`20_sortie_attention/code/gpt2_vsel.py`](20_sortie_attention/code/gpt2_vsel.py) et [`gpt2_vsel3.py`](20_sortie_attention/code/gpt2_vsel3.py) — reclassement des blocs par alignement de valeur, plus controles (direction aleatoire, direction opposee) et protection de l'argmax.
- [`20_sortie_attention/resultats/gpt2_vsel.txt`](20_sortie_attention/resultats/gpt2_vsel.txt) — 15 configurations : aucune variante sensible a la valeur ne bat le max cle seule ; l'ordre correct < aleatoire < anti montre que le signal existe.
- [`20_sortie_attention/resultats/gpt2_vsel3.txt`](20_sortie_attention/resultats/gpt2_vsel3.txt) — 9 configurations : proteger l'argmax du selecteur redonne les chiffres au chiffre pres, donc le pic deloge n'est pas l'argmax et le degat est dans le remplissage du budget.
- [`17_papier/asp.tex`](17_papier/asp.tex) — papier a jour (16 pages, 744 218 octets) : paragraphe du banc de recuperation etendu au test de valeur.\n
## Budget adaptatif (2026-09-09) — refutation

- [`20_sortie_attention/code/gpt2_tau.py`](20_sortie_attention/code/gpt2_tau.py) — selection par seuil tau * meilleur score, plafonnee a m, avec compteur de blocs lus (corrige du facteur nb).
- [`20_sortie_attention/resultats/gpt2_tau.txt`](20_sortie_attention/resultats/gpt2_tau.txt) — 18 configurations : a octets egaux le rang bat le seuil (a m=24, tau=0,5 economise 51 % des blocs et coute +0,72 nat, alors que tronquer le rang de 24 a 8 blocs coute +0,66 nat).\n
## Index a deux etages (2026-09-09) — resultat positif

- [`20_sortie_attention/code/gpt2_two.py`](20_sortie_attention/code/gpt2_two.py) et variantes `gpt2_fine.py`, `gpt2_finesum.py`, `gpt2_finenorm.py` — etage 1 = top-2m candidats par l'index grossier d'=8, etage 2 = re-classement des candidats.
- [`20_sortie_attention/resultats/gpt2_two.txt`](20_sortie_attention/resultats/gpt2_two.txt) — le re-classement EXACT atteint l'oracle : +0,01128 nat a m=16 (contre +0,22553), +0,24636 a m=8 (contre +0,69516) ; controle aleatoire catastrophique (+1,15 a +1,82).
- [`20_sortie_attention/resultats/gpt2_fine.txt`](20_sortie_attention/resultats/gpt2_fine.txt), [`gpt2_finesum.txt`](20_sortie_attention/resultats/gpt2_finesum.txt), [`gpt2_finenorm.txt`](20_sortie_attention/resultats/gpt2_finenorm.txt) — aucun score compact teste ne reproduit le gain ; a m=24 le cosinus 32d a S=0,949 et 99,9 % de pic pour +0,358 nat, contre +0,038 nat pour la reference (S=0,913, 97,1 %).\n
## Diagnostic de la base (2026-09-09)

- [`20_sortie_attention/code/gpt2_orbasis.py`](20_sortie_attention/code/gpt2_orbasis.py) — re-classement des 2m candidats avec la base ORACLE (32 vecteurs singuliers dominants du K courant, par tete).
- [`20_sortie_attention/resultats/gpt2_orbasis.txt`](20_sortie_attention/resultats/gpt2_orbasis.txt) — le defaut est le RANG, pas la base : +0,166 nat a 37,5 % et +0,282 a 50 % (contre +0,011 et +0,008 pour l'exact), et a 50 % c'est 7x pire que de ne pas re-classer, malgre 99,9 % de pic et 95,4 % de masse retenue.\n
## Deuxieme etage : precision et nature du score (2026-09-09)

- [`20_sortie_attention/code/gpt2_quant.py`](20_sortie_attention/code/gpt2_quant.py), [`gpt2_quantp.py`](20_sortie_attention/code/gpt2_quantp.py), [`gpt2_norm.py`](20_sortie_attention/code/gpt2_norm.py) — re-classement quantifie (8/4 bits, echelle max ou ecretage au percentile) et selecteur sans requete.
- [`20_sortie_attention/resultats/gpt2_quant.txt`](20_sortie_attention/resultats/gpt2_quant.txt) — 8 bits a echelle max : +0,179 nat a 37,5 % (contre +0,011 exact) malgre 0,8 % d'erreur par element ; a m=8 les 8 et 4 bits valent l'exact.
- [`20_sortie_attention/resultats/gpt2_quantp.txt`](20_sortie_attention/resultats/gpt2_quantp.txt) — ecretage au 99,9e percentile : catastrophique (+1,21 nat, pic 66,5 %), car les cles de grande norme portent l'ordre.
- [`20_sortie_attention/resultats/gpt2_norm.txt`](20_sortie_attention/resultats/gpt2_norm.txt) — score sans requete (norme des cles) : +1,04 nat a 37,5 %, pic 31,6 % ; la famille top-norm de Qwen3-8B ne se transporte pas.\n
## Generalisation a un deuxieme modele (2026-09-09)

- [`20_sortie_attention/code/bench_llama.py`](20_sortie_attention/code/bench_llama.py) — portage du banc a deux etages sur une architecture Llama via hooks (capture Q/K post-RoPE, remplacement de la sortie d'attention), sans patch de forward.
- [`20_sortie_attention/resultats/bench_smol135.txt`](20_sortie_attention/resultats/bench_smol135.txt) — SmolLM2-135M, m=16, W=32 : grossier +0,32867 ; exact +0,05096 ; masse +0,05258 ; 8 bits +0,04242 ; norme +1,61627 (pic 23,0 %). Le gain du deuxieme etage et l'echec du score sans requete REPRODUISENT ; la quantification, NON (elle est au niveau de l'exact ici, 16x pire sur GPT-2).
- [`20_sortie_attention/resultats/diag_ratio.txt`](20_sortie_attention/resultats/diag_ratio.txt) — ratio max|k|/p99,9|k| par tete : GPT-2 1,07, SmolLM2 1,18, Qwen2.5-0.5B 1,18. L'hypothese de l'aberration d'echelle est REFUTEE.\n
## Mecanisme de la divergence entre modeles (2026-09-09)

- [`20_sortie_attention/code/gpt2_swap_only.py`](20_sortie_attention/code/gpt2_swap_only.py), [`bench_llama_swap_only.py`](20_sortie_attention/code/bench_llama_swap_only.py) — courbe de sensibilite Delta(k) : k blocs de l'ensemble exact remplaces par des candidats tires au hasard.
- [`20_sortie_attention/resultats/gpt2_swap.txt`](20_sortie_attention/resultats/gpt2_swap.txt) — GPT-2 : k=1 +0,167 nat, k=2 +0,286, k=4 +0,677.
- [`20_sortie_attention/resultats/bench_smol135_swap.txt`](20_sortie_attention/resultats/bench_smol135_swap.txt) — SmolLM2 : k=1 +0,010 nat, k=2 +0,026, k=4 +0,147. Le paysage est 16 a 52 fois plus raide sur GPT-2, ce qui explique pourquoi la quantification y est couteuse et gratuite sur SmolLM2.\n- [`20_sortie_attention/resultats/bench_qwen05_swap.txt`](20_sortie_attention/resultats/bench_qwen05_swap.txt) — Qwen2.5-0.5B (3e modele) : k=1 +0,047 (PLAT), 8 bits +0,046 vs exact +0,048 (quantification gratuite), norme +1,811 (pic 30,7 %). La regle Delta(1) est confirmee sur trois modeles.

## Deuxieme etage en contexte long (2026-09-09)

- [`20_sortie_attention/code/echelle_two.py`](20_sortie_attention/code/echelle_two.py) — banc long-contexte instrumente : candidats top-2m par score d'=8, re-classement exact ou 8 bits, courbe Delta(k).
- [`20_sortie_attention/resultats/echelle_two.txt`](20_sortie_attention/resultats/echelle_two.txt) — T=1024, 25 % de cles, SmolLM2 : dense 2,75635 ; grossier +0,05909 ; exact +0,01178 ; 8 bits +0,01128 ; aleatoire +0,69392 ; swap k=1/2/4 = +0,02703/+0,03724/+0,08519. Gain du deuxieme etage divise par 5, quantification gratuite, Delta(1)=0,015 conforme a la regle.
- Trois points de validation long-contexte (meme fichier `echelle_two.txt`) : T=1024/off 0 (grossier +0,05909, exact +0,01178, 8 bits +0,01128, Delta(1)=0,01525) ; T=1024/off 200k (grossier +0,01347, exact -0,00497 BAT le dense, 8 bits -0,00503, 4 bits -0,00452, Delta(1)=0,01827) ; T=2048/off 0 (grossier +0,00952, exact +0,00570, 8 bits +0,00583, 4 bits +0,00611, Delta(1)=0,01358). Delta(1) reste dans [0,014 ; 0,018] et la quantification 8 et 4 bits est gratuite partout.
- [`20_sortie_attention/code/bilan_octets.py`](20_sortie_attention/code/bilan_octets.py) et [`resultats/bilan_octets.txt`](20_sortie_attention/resultats/bilan_octets.txt) — audit octets : l'etage 2 a 8 bits lit 1,000x les octets de l'index a un etage (520/520), a 4 bits 0,877x, l'exact 1,246-1,330x. Corrige les chiffres non audites de 1,05x et 8 % de K.
- Quatrieme point (nouvelle architecture) : Qwen2.5-0.5B, T=1024, off 0, D=64 — dense 2,59795 ; grossier +0,00767 ; exact +0,00072 (facteur 10,7) ; 8 bits +0,00204 ; 4 bits +0,00016 ; Delta(1)=-0,00245. La configuration byte-neutre survit au changement de modele.
- [`20_sortie_attention/code/reconciliation_octets.py`](20_sortie_attention/code/reconciliation_octets.py) et [`resultats/reconciliation_octets.txt`](20_sortie_attention/resultats/reconciliation_octets.txt) — reconciliation des deux comptabilites d'octets : les ratios 13,47/17,66/20,90/23,01 du harness e2e sont reproduits au centieme ; l'ecart avec le banc qualite (3,94x) vient du budget (1,56 % des cles contre 25 %) et du mecanisme de l'etage fin (resumes C2 contre cles brutes, facteur 16).
- Configs `r2`/`r1` du banc `echelle_two.py` (re-classement sur resumes de segment) : a octets egaux avec `q8h`/`q4h`, le resume egale la quantification a 520 unites (+0,01085 vs +0,01128) mais perd 2,6x a 456 (+0,03039 vs +0,01168) sur SmolLM2 ; sur Qwen2.5-0.5B tout reste dans +/-0,002 nat. La moyenne detruit le pic, la quantification preserve l'ordre.
- La penalite de dilution suit Delta(coarse) : 2,6x a 25 % quand le grossier laisse +0,059 nat (off 0), nulle a l'offset 200k (+0,013), au budget 50 % (+0,019) et sur Qwen2.5-0.5B (+0,008). Le mecanisme d'etage 2 doit etre choisi sur Delta(coarse).
- Echelle en budget de la penalite de dilution (SmolLM2, T=1024) : +0,112 nat a 6,25 % (r1 pire que le grossier), +0,044 a 12,5 %, +0,019 a 25 %, nulle a 50 %. Le harness (1,56 % des cles) est sous le point ou la moyenne fait du degat.
- Part du gain d'etage 2 recuperee par budget (SmolLM2, T=1024) : quantification de cles brutes 78-104 % a TOUS les budgets ; resumes de segment 96 % -> 35 % (r2) et 104 % -> 16 % (r1) quand le budget tombe de 50 % a 1,56 % (fraction du harness).
- Test a la dilution exacte du harness (Lb=64, 8 resumes de 8 cles par bloc, octets egaux 16 u) : ns8 recupere 10,9 % du gain d'etage 2, q2h 76,4 % ; ns16 (4 cles/resume) 55,5 %. Le max sur 8 resumes ne restaure pas le rang.
- Portabilite sur Qwen2.5-0.5B : a 1,56 %, q8h 89,9 %, q4h 77,9 %, r2 40,3 %, r1 -144 % (pire que le grossier de 0,27 nat) ; a 6,25 %, r2 117,3 % (meilleur) et r1 -100,6 %. Le nombre de cles par resume gouverne : 4 cles collapsent des 6,25 %, 8 cles (harness) ne recuperent que 10,9 %.
- Plancher de precision a la fraction du harness (1,56 %, SmolLM2) : q4h 90,5 %, q2h 82,4 %, q8h 78,4 %, q1h -53,4 % (pire que ne pas re-classer). Deux bits = 16x de compression, moins couteux et 8x meilleur que des resumes de 8 cles.
- Plancher de 2 bits INDEPENDANT du budget (SmolLM2, part du gain d'etage 2) : 1 bit est toujours pire que le grossier (-53 % a 1,56 %, -52 % a 6,25 %, -14 % a 25 %) ; 2 bits 65-82 %, 4 bits 90-101 %.
- Plancher de 2 bits confirme sur la 2e architecture (Qwen2.5-0.5B, 1,56 %) : q8h 89,9 %, q4h 77,9 %, q2h 53,0 %, q1h -554,7 % (1 bit coute +1,71 nat contre +0,68 pour le grossier). Au-dessus du plancher la recuperation depend du modele.
- Le remede est STRICTEMENT DOMINANT (comptage analytique) : remplacer le terme m*r2 (resumes C2) par k*Lb*p (cles brutes quantifiees) porte le ratio d'octets de 13,47/17,66/20,90/23,01 a 14,84/20,08/24,38/27,30 (2 bits) tout en recuperant 76 % du gain d'etage 2 au lieu de 10,9 %.
- Premier etage (Lb=64, W=8, m=1) : les resumes moyens NE sont PAS domines — ns4 (4 resumes de 16 cles, 4 elem/bloc) +1,33340 contre coarse (index PCA max, 4 elem/bloc) +1,35457 ; echelle monotone ns1 +1,82862, ns2 +1,47571. La dilution n'est fatale qu'au re-classement fin.
- Selecteur complet : les resumes du harness pesent 31,6/41,4/49,0/53,9 % de ses octets (N=16k/32k/64k/131k) ; l'architecture du banc (index PCA d'=8 8 bits + cles brutes 2 bits) economise 29,9/39,2/46,4/51,1 % et porte le ratio de 13,47/17,66/20,90/23,01 a 19,23/29,05/39,01/47,08.
- Premier etage a octets egaux (Lb=64, W=8, m=1) : le score a resumes moyens du harness est DOMINE de 2,2-2,4 nat par l'index PCA max — 4 elem/bloc +3,73223 vs +1,35457 ; 2 elem/bloc +3,83728 vs +1,61479 ; 1 elem/bloc +3,87465 vs +1,60282.
- Controle de normalisation (meme point, 4 elem/bloc) : resumes normalises +3,00339 contre +1,35457 pour l'index PCA — la dilution n'est pas un artefact d'echelle ; le harness n'utilise aucune normalisation.
- Portabilite du premier etage (meme protocole, 4 elem/bloc) : ecart resumes/index de 2,38 nat sur SmolLM2-135M et 3,13 nat sur Qwen2.5-0.5B ; resumes normalises 1,65 et 1,23 nat d'ecart.
- Second etage, portabilite (Lb=64, candidats PCA) : sur SmolLM2 ns4 +1,6 % / ns2 -24,4 % / ns1 -96,3 % du gain ; sur Qwen ns2 -15,6 % / ns4 -42,9 % / ns1 -74,7 % ; exact 100 % sur les deux.
- Normalisation au second etage (cns*, Lb=64) : elle degrade partout — SmolLM2 -66,3 / -60,6 / -50,1 % du gain, Qwen -197,2 / -134,6 / -156,0 % — alors qu'au premier etage elle gagnait 0,73 / 1,90 nat.
- Normalisation au 2e etage, controle a Lb=4 : nuisible encore plus qu'a Lb=64 (-0,91/-0,94 nat SmolLM2, -1,38/-1,75 Qwen) ; marge de l'etage 2 bornee par le rappel de l'etage 1 (0,007 nat SmolLM2, 0,070 Qwen a Lb=4).

---

## ▶ Prochaine étape : au-delà d'ASP

**Bilan pour trancher** : ASP (sélection de blocs à deux passes) a été étudié en profondeur
— théorie ([12_poc](12_poc/RAPPORT_POC.md)), papier complet ([17_papier/asp.pdf](17_papier/asp.pdf)),
recherche fine sur le sélecteur (journal ci-dessus), et 3 sessions GPU réelles
([Axe 21](21_rtx4090_reel/RAPPORT_RTX4090.md)). Le verdict est stable et reproduit : ASP
économise des octets mais ne devient pas plus rapide, y compris après correction de deux
défauts distincts identifiés en session 3. La piste est considérée **close** pour ce projet.

**Ce qui reste exploitable d'ASP pour la suite** (résultats transférables, indépendants
d'ASP lui-même) :
- La loi octets-précision ([19_loi_octets](19_loi_octets/RAPPORT_OCTETS.md)) : la
  quantification bat toujours les résumés par centroïdes/moyennes à budget égal — probablement
  vrai pour toute technique de compression du KV cache, pas seulement ASP.
- Le mécanisme sink + fond plat de l'attention hors-fenêtre (§ Axe 20) — structure générale
  de l'attention, utile pour n'importe quelle méthode de sélection/éviction.
- Les 4 architectures/GPU de bancs mémoire (`16_gpu/`) — réutilisables pour tester le
  comportement mémoire de n'importe quelle autre technique.

**La recherche s'oriente maintenant vers d'autres techniques d'optimisation de
l'inférence** (au-delà de la sélection de blocs par deux passes). Voir le corpus déjà
constitué pour des pistes non explorées expérimentalement : quantification des poids
([08_quantization_poids](08_quantization_poids/)), MoE ([09_moe_inference](09_moe_inference/)),
serving/désagrégation ([05_serving_systems](05_serving_systems/)), décodage spéculatif
([04_speculative_decoding](04_speculative_decoding/)), matériel
([11_materiel](11_materiel/)). Nouvel axe numéroté à créer (22_...) quand une direction
sera choisie.
