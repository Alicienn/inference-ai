# Axe 2 — Compression du KV cache : quantification, éviction, allocation

## 1. Trois familles, une hiérarchie de gains mal ordonnée

Le corpus couvre trois leviers, souvent présentés comme concurrents alors qu'ils sont orthogonaux :

| Levier | Principe | Gain typique annoncé | Papiers du corpus |
|---|---|---|---|
| **Réduction architecturale** | Moins d'état à stocker (MLA, MQA, compression CSA/HCA) | 10–50× | V3 (MLA), V4 (CSA/HCA) |
| **Quantification** | Moins de bits par élément | 2–8× | TurboQuant, RDKV, Block-GTQ, INT8/INT4 |
| **Éviction / sélection** | Moins d'éléments conservés | 2–10× | revue 2508.06297, ARKV |

**[FAIT]** Les gains se composent : V4-Flash cumule compression architecturale (m=4/m'=128),
format mixte (BF16 sur RoPE, FP8 ailleurs) et sélection (top-k=512), pour atteindre **1,87 % d'une
baseline BF16 GQA8** — soit 53× (mesuré, cf. `model_inference_cost.py`).

**[INFÉRENCE]** L'ordre d'importance est inversé par rapport à l'attention que leur porte la
littérature. Le levier architectural rapporte le plus (14× de V3.2 à V4-Flash), la quantification
vient ensuite (2×), et l'allocation fine de bits — quasi absente des discussions — vaut encore
~2 bits (§3). Or le volume de publications est à peu près l'inverse.

## 2. La métrique manquante : résident vs lu

**[FAIT]** Voir [`SYNTHESE.md`](SYNTHESE.md) §3. Tous les papiers de ce corpus rapportent une
réduction de **taille résidente**. La latence de décodage dépend des **octets lus par token**. Les
deux coïncident en dense, divergent en creux (facteur 5,9× à 7,8× dans mes mesures).

**[INFÉRENCE]** Conséquence pratique, importante pour évaluer une méthode d'éviction : une méthode
qui doit **scanner tout le cache** pour décider quoi évincer (scoring global à chaque pas) réduit
le résident sans réduire le lu — donc **n'améliore pas la latence**, seulement la capacité. Il
faut exiger des papiers d'éviction qu'ils déclarent le coût de leur propre politique de décision.
C'est exactement l'erreur que DSA commet à grande échelle (§ axe 1 : l'indexeur coûte 90 % du
budget).

## 3. Allocation de bits : le gain sous-estimé — vérification indépendante

Le papier **Block-GTQ** ([2606.24033](https://arxiv.org/abs/2606.24033)) pose que le logit
d'attention RoPE se décompose en une somme sur des blocs de fréquence 2D, et que leur énergie est
« fortement inégale », justifiant une allocation de bits par bloc plutôt qu'uniforme, avec la loi
de débit-distorsion MSE ∝ 4^(−b) de TurboQuant-MSE.

J'ai testé cette prémisse sur les activations réelles de **SmolLM2-135M** (30 couches, GQA 9/3,
head_dim 64 → 32 blocs de fréquence, RoPE base 10⁴).

**[FAIT] Prémisse confirmée.** Sur 90 profils (couche × tête KV), étendue max/médiane de l'énergie
par bloc = **793× en médiane**, p90 = 2 688×, max = 4 873×. Soit ~3 ordres de grandeur, ce qui
correspond bien à la description du papier.

**[FAIT] Structure supplémentaire.** L'énergie n'est pas répartie au hasard entre fréquences :
blocs **basse fréquence** (i≥16, rotation lente) = 86,8 en moyenne normalisée, contre 2,49 pour
les hautes fréquences → **ratio 34,9×**. La concentration est marquée sur les blocs 22-25
(jusqu'à 528× la médiane).

**[INFÉRENCE]** Le papier présente le profil comme irrégulier tête par tête. Mes mesures montrent
en plus un **gradient systématique en fréquence**. Si ce gradient se confirme sur d'autres
modèles, une heuristique statique « plus de bits aux basses fréquences » capturerait une grande
partie du gain **sans calibration**, ce qui simplifierait beaucoup le déploiement par rapport à
l'allocateur glouton par tête de Block-GTQ. Testable immédiatement (Q4).

**[FAIT] L'allocation guidée bat l'uniforme**, quantification appliquée *après* RoPE (comme dans
les systèmes réels), erreur relative sur les logits et KL de la distribution d'attention :

| bits moyens | err. uniforme | err. allouée | gain | KL uniforme | KL alloué |
|---:|---:|---:|---:|---:|---:|
| 2 | 0,2397 | **0,0424** | 82,3 % | 0,8375 | **0,0225** |
| 3 | 0,0798 | **0,0212** | 73,5 % | 0,1440 | **0,0054** |
| 4 | 0,0341 | **0,0088** | 74,2 % | 0,0315 | **0,0010** |
| 6 | 0,0077 | **0,0018** | 76,3 % | 0,0018 | **0,00004** |
| 8 | 0,0019 | **0,00044** | 76,5 % | 0,00011 | **~0** |

**[INFÉRENCE]** En interpolant sur la courbe uniforme, l'allocation guidée vaut **1,4 à 2 bits
gratuits** : un cache à 3 bits alloués bat un cache à 4 bits uniformes. Soit 25 à 40 % de mémoire
économisée à qualité constante — davantage que ce que rapporte un changement de format.

## 4. Lecture critique du format mixte de DeepSeek-V4

**[FAIT]** V4 stocke les 64 dimensions RoPE en BF16 et les autres en FP8 (§2.3.4), « ce qui réduit
la taille du cache de près de moitié par rapport au BF16 pur ». L'indexeur calcule en FP4.

**[INFÉRENCE]** C'est une allocation **binaire** — un cas très grossier de l'allocation par blocs.
Mes mesures indiquent que l'écart d'énergie décisif se joue **entre bandes de fréquence à
l'intérieur même de la partie RoPE** (gradient 35×), gradient qu'un schéma à deux niveaux ne peut
pas capturer. Il reste donc plausiblement du gain sur la table pour V4.

**[HYPOTHÈSE]** Appliquer Block-GTQ *à l'intérieur* des 64 dimensions RoPE de V4 (au lieu d'un
BF16 uniforme) permettrait de descendre ces dimensions à ~4 bits moyens à qualité égale, soit
~30 % de KV cache en moins sur la partie RoPE. Sur V4-Flash cette partie pèse 128 octets sur 576
par entrée CSA (22 %) → gain net attendu ~7 % du cache total. Modeste mais gratuit. **Non testé.**

**[INFÉRENCE] Un angle mort de la littérature.** La littérature « architecture » (DeepSeek) et la
littérature « quantification » (Block-GTQ, TurboQuant, RDKV) **ne se citent pas**, alors qu'elles
optimisent le même objet. DeepSeek choisit ses précisions à la main ; les quantificateurs
travaillent sur des architectures GQA standard et **ignorent MLA/CSA**, dont l'état latent partagé
change complètement la structure du problème (une entrée sert à la fois de clé et de valeur —
l'erreur de quantification s'y propage deux fois). Aucun papier du corpus n'étudie la
quantification d'un cache **latent partagé**. Voir Q3.

## 5. TurboQuant et la ligne « rotation + quantification scalaire »

**[FAIT]** TurboQuant ([2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) : rotations
aléatoires (PolarQuant) puis quantification scalaire optimale, plus un correcteur d'erreur 1-bit
QJL ; 6× de compression du cache et 8× d'accélération de l'attention.

**[INFÉRENCE]** La rotation aléatoire fonctionne parce qu'elle **égalise l'énergie entre
dimensions** (elle détruit les canaux aberrants). C'est en tension directe avec Block-GTQ, qui
**exploite** l'inégalité d'énergie. Les deux ne se composent donc pas naïvement : appliquer une
rotation aléatoire globale détruirait la structure par blocs RoPE que Block-GTQ utilise. Block-GTQ
en est conscient (il réutilise l'encodeur TQ-MSE *par groupe de blocs de même largeur*), mais
l'articulation mérite d'être étudiée. Question Q4.

## 6. Limites

- Un seul modèle (SmolLM2-135M) pour les mesures d'énergie RoPE. Le papier Block-GTQ valide sur
  dix modèles ; ma vérification confirme la prémisse mais ne teste pas sa généralité.
- Quantificateur uniforme symétrique avec échelle par (token, bloc) — plus simple que TQ-MSE. Mes
  gains relatifs uniforme→alloué sont donc probablement **conservateurs**.
- Métriques intermédiaires (erreur de logit, KL d'attention), pas de perplexité ni de benchmark.


## Addenda v2 — compléments vérifiés

- Date : 2026-08-30 · Corpus : 31 papiers arXiv
- [FAIT] **TurboQuant** [2504.19874] (Google, ICLR 2026) : rotations aléatoires (Hadamard incohérentes) + quant scalaire (allocation par token, au bit près) + correcteur 1-bit (dans l'esprit QJL) ; démontre une borne inférieure de distorsion (Thm 3) atteinte par sa méthode. **FP8 robuste même à 4× compression** ; compression **≥4.5×** mesurée sur LongBench-V1 (Llama-3.1-8B) ; ratio 0.25 testé. Les « 6× / 8× attention plus rapide » d'INDEX.md ne sont pas retrouvés tels quels dans le texte — à re-vérifier sur la version OpenReview. [FAIT pour ≥4.5×, INFÉRENCE pour 6×/8×]
- [FAIT] **Block-GTQ** [2606.24033] (RoPE-aware) : allocation par blocs consciente de RoPE. K3V3 (3 bits par composante K et V) = **3.24× compression** à 128K, qualité comparable fp16 ; **1.34× décodage plus rapide** que FlashAttention-2 fp16 à 128K ; 296: coût d'un bit mal placé b=3 vs b=4 : facteur ~4×. Améliore TurboQuant sur les composantes RoPE (l'orthogonalité d'Hadamard ne préserve pas la structure RoPE).
- [FAIT] **Quant-vs-Rank** [2604.11501] : à budget mémoire égal, **INT4 domine la réduction de rang** — INT4 joint K+V = +0.18 PPL à 75% réduction (Mistral 7B, Table 3 ; +0.23 dans un autre réglage) contre +34.77 PPL pour rank-32 ; même rank-64 INT8 échoue à égaler INT4 ; les bases d'éviction (H2O/SnapKV-like) sont équivalentes (spread <0.4 PPL).
- [FAIT] **FlashAttention-3** [2407.08608] : 740 TFLOPs/s FP16 (75% util. H100) ; ~1.2 PFLOPs/s FP8 ; FP8 2.6× moins d'erreur qu'une baseline FP8 naïve. Le kernel dense de référence — tout gain « attention » des méthodes ci-dessus se mesure contre lui.
- [FAIT] **RDKV** [2605.08317] : formulation débit-distorsion. Théorème central (Thm 3.3) : **l'éviction est le régime b=0 de la quantization** — les deux opérations sont deux régimes d'un même problème d'allocation de bits. Prop 3.1/3.2 : poids d'éviction d'un token = a_{τ,t} (poids d'attention), poids d'un canal = rank-one. Pipeline 3 étapes + layout packed mixed-bit. Chiffres : **97.81% de l'accuracy LongBench avec 2.48% de rétention** ; latence/token ~18 ms flat de 8K→128K (vs 26→82 ms FullKV) = **4.5× décodage** ; cache **1.9×** réduit à 128K.
- [FAIT] **SpargeAttention2** [2602.13515] : masquage hybride **top-k + top-p** (sélection par blocs, seuil p adaptatif par distribution) + distillation pour entraîner les composants ; **16.2× speedup attention** vs dense, **4.7× génération end-to-end**, 1.4× plus rapide que son prédécesseur SLA, >4× que VSA. C'est l'évolution du SpargeAttn original (remplace le seuil léger par top-p).
- [FAIT] **DashAttention** [2605.18753] : attention hiérarchique creuse **différentiable et adaptative**. Critique frontale du top-k fixe : « fixed sparsity budgets allocate disproportionate compute to over-attended and under-attended regions ». **3.36× vs FlashAttention-3** à 96K (s=93.75% : 93.75% du compute économisé), 1.35× vs InfLLMv2 ; plage 1.34-3.09×.
- [FAIT] **Review KV-Cache Compression** [2508.06297] : taxonomie (quantization, éviction/réduction, merging, low-rank, hybrid) ; ~35+ méthodes cartographiées ; le champ est très actif mais les comparaisons inter-méthodes restent fragmentaires.
- - **Quantization** : à budget égal, elle domine (INT4 joint K+V +0.18 PPL vs rank-32 +34.77) [2604.11501]. La question n'est plus « faut-il quantizer » mais « quelle allocation de bits ».
- - **Éviction** : nécessaire au-delà d'un ratio (la quantization seule plafonne ~4× pour rester near-lossless sur Llama-3.1-8B) [2504.19874] ; RDKV montre qu'elle est le régime b=0 de la même allocation [2605.08317].
- | TurboQuant | 2504.19874 | quant. | ≥4.5× (0.25 ratio) | near-lossless @4× FP8 | (6×/8× à vérifier) | rotations incohérentes nécessaires | A100/H100 |
- | Block-GTQ | 2606.24033 | quant. RoPE-aware | 3.24× (K3V3) @128K | ~fp16 | 1.34× decode @128K | structure RoPE ; allocation par blocs | — |
- | INT4 K+V | 2604.11501 | quant. | 75% réduction (INT4) | +0.18 PPL (Table 3) | — | domine rank-32 au même budget | — |
- | RDKV | 2605.08317 | éviction+quant. jointes | 2.48% rétention | 97.81% acc LongBench | 4.5× decode, 1.9× cache | distorsion TV de l'attention ; pipeline 3 étapes | — |
- | SpargeAttention2 | 2602.13515 | sparsité distillée | top-k+p par bloc | ≈ acc dense (distill) | 16.2× attention, 4.7× E2E | entraînement distillation | H100 |
- | DashAttention | 2605.18753 | sparsité adaptative | s=93.75% @96K | ≈ dense | 3.36× vs FA-3, 1.35× vs InfLLMv2 | adaptatif vs top-k fixe | H100 |
- | FlashAttention-3 | 2407.08608 | kernel dense | — | réf. | 740 TF FP16 / 1.2 PF FP8 | Hopper, interleave softmax | H100 |
- | MLA→CSA/HCA | axe 1 | architecture | ~2% GQA8 @1M | entraînée | 10×/27× FLOPs | coûte un pretraining | H800 |
- 2. **Top-k fixe vs adaptatif** : NSA/DSA (fixe) vs SpargeAttn2/DashAttention (adaptatif). DashAttention expose la critique ; aucune comparaison croisée NSA-64k vs Dash-96k dans le corpus. [FAIT]
- 3. **INT4 domine le rang — mais sur quel modèle ?** [2604.11501] sur Mistral 7B : rangée applicabilité à MoE 600B/attention partagée non établie. [FAIT pour la limite ; INFÉRENCE pour l'impact]
- 4. **V4 (FP8+BF16 mixte) n'adopte PAS INT4/INT2 pour le KV** : V4 reste à FP8 pour le KV hors-RoPE (BF16 pour les 64 dims RoPE) — suggère que INT4 cache-quality était jugé risqué pour 1M tokens, ou que le FP4 (indexer) + FP8 (KV) suffit au budget visé. [FAIT pour le choix ; INFÉRENCE pour la raison]
- | Court contexte (<32K), grand batch, GPU H100+ | FP8 simple (FA-3) | Le cache tient en mémoire ; le gain de la quantization est secondaire ; FA-3 FP8 ~1.2 PFLOPs/s [2407.08608] |
- | Long contexte (32K-256K), mémoire HBM limitée | INT4/INT8 avec rotations incohérentes (TurboQuant) ou RoPE-aware (Block-GTQ) | TQ ≥4.5× near-lossless [2504.19874] ; Block-GTQ 3.24× @128K + 1.34× decode [2606.24033] |
- | Très long (≥512K), agentic (requêtes longues) | Architecture native (CSA/HCA) + FP8/BF16 mixte + offloading LSA | Le post-hoc plafonne : INT4 domine rang mais reste O(L) stockage ; seul le natif casse le O(L·k) FLOPs [2606.19348] |
- | Agentique à forte réutilisation de préfixe | Éviction RDKV + cache de préfixe (Mooncake) | RDKV : 97.81% acc à 2.48% rétention [2605.08317] ; Mooncake +525% throughput via réutilisation [2407.00079] |
- | Batch 1-8 (latence), modèle dense | Sparsité entraînable (SpargeAttn2/DashAttention) | 4.7× E2E [2602.13515] ; 3.36× vs FA-3 [2605.18753] |
- | Batch élevé (throughput) | Quantization plutôt que sparsité | La sparsité spéculative s'effondre au batch (axe 3, EAGLE-3 1.81×→1.32×) ; la quantization reste |
- - L'éviction et la quantization sont un continuum (RDKV Thm 3.3) : allouer 0 bit = éviction. Les traiter séparément est une erreur d'ingénierie.
- 1. **Frontière quantization/éviction** : RDKV unifie les deux, mais aucun papier ne mesure la frontière pratique qualité/rétention par tâche. Protocole : sweep budget de bits par token {0,2,3,4,8} × {RDKV, Block-GTQ, TQ} sur LongBench-v2 + MRCR à 512K.
- 3. **Loi d'échelle de la distorsion en longueur** : la distorsion de quantization observée à 8K/128K (TQ, Block-GTQ) tient-elle à 1M ? [HYPOTHÈSE : non — l'erreur se concentre sur les tokens distants]
- 4. **INT4 sur MoE géants** : les conclusions de 2604.11501 (Mistral 7B) se généralisent-elles aux MoE 284B+ avec attention partagée (MQA/MLA) ? L'attention partagée devrait réduire la redondance exploitable par les rotations.

## 7. Références du corpus

- `02_surveys/Review_KV-Cache-Compression_2508.06297.pdf` — panorama
- `03_attention_kv_cache/TurboQuant_KV-Quantization_2504.19874.pdf`
- `03_attention_kv_cache/RoPE-Aware-Bit-Allocation-KV_2606.24033.pdf` — Block-GTQ
- `03_attention_kv_cache/RDKV_Rate-Distortion-KV_2605.08317.pdf` — éviction + quantification jointes
- `03_attention_kv_cache/Quantization-vs-Rank-Reduction-KV_2604.11501.pdf`
- Code : [`_tools/exp_B_rope_bit_allocation.py`](../_tools/exp_B_rope_bit_allocation.py)
