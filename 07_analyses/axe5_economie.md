# Axe 5 — Vue économique : coût par token servi

Date : 2026-08-30 · Corpus : 31 papiers arXiv

---

## 1. Le coût par token : de quoi parle-t-on ?

Le coût de service d'un token se décompose en :
- **FLOPs/token** (compute) — dépend de l'architecture (paramètres actifs, attention) ;
- **Mémoire** (poids + KV cache) — dépend de la taille du modèle, de la compression et de la longueur de contexte ;
- **Bande passante** (HBM pour le décodage, réseau pour la désagrégation) ;
- **Coût d'exploitation** ($/h GPU).

Le corpus fournit des chiffres réels (DeepSeek publie ses coûts H800 à $2/h) et des ratios architecturaux. Ce qui manque : une comparaison normalisée inter-papiers (chaque papier mesure sur son propre matériel/benchmark). Ce document consolide ce qui est chiffré et signale explicitement ce qui ne l'est pas.

---

## 2. Chiffres consolidés du corpus

### 2.1 Coûts réels DeepSeek (les seuls $/M tokens du corpus)

[FAIT] **V3.2/DSA** [2512.02556, Fig. 3] — coûts réels sur H800 ($2/h/GPU) :
- Prefill 128K : ~0.7 → ~0.2 $/M tokens avec DSA (facteur ~3.5×) ;
- Decode 128K : 2.4 → ~0.8 $/M tokens (facteur ~3×).
Ces coûts incluent l'énergie ? Non précisé — ce sont des coûts d'amortissement GPU (location). [FAIT pour les valeurs ; lacune sur la méthode exacte]

[FAIT] **V4** [2606.19348, §1 + Fig. 1] — ratios vs V3.2 à 1M tokens :
- V4-Pro : 27% FLOPs/token, 10% KV cache ;
- V4-Flash : 10% FLOPs/token, 7% KV cache.
Mais ces ratios confondent architecture et taille de modèle (V4-Flash 13B actifs vs V3.2 37B actifs ; V4-Pro 49B actifs). La comparaison architecture-seule (à paramètres actifs proches : V4-Pro vs V3.2) donne **~3.7× FLOPs et ~10× KV**. [INFÉRENCE]

[FAIT] **V3** [2412.19437] : coût total d'entraînement 5.576 M$ (2.788 M$/H800×184k) — coût d'entraînement, pas d'inférence, mais utile pour l'ordre de grandeur du capital mobilisé.

### 2.2 Ratios de performance par technique (gains transposables en $)

| Technique | Gain mesuré | Source | Conversion en coût d'inférence [INFÉRENCE] |
|---|---|---|---|
| DSA (fine-grain sparse) | 0.7→0.2 $/M prefill, 2.4→0.8 $/M decode @128K | [2512.02556 Fig.3] | ×3 direct (coût GPU mesuré) |
| CSA/HCA (V4-Pro vs V3.2) | 27% FLOPs, 10% KV @1M | [2606.19348 §1] | ×~3.7 FLOPs, ×10 KV (architecture-seule) |
| LSA (indexer KV offload) | 0.30× compute/token @1M, 13.5% GPU KV, 2.8× throughput | [2606.09079 Table 1] | ×3 compute ; ×7.4 KV GPU |
| RDKV (éviction+quant) | 4.5× decode, 97.81% acc à 2.48% rétention | [2605.08317] | ×4.5 sur la part decode |
| TurboQuant (quant) | ≥4.5× compression near-lossless | [2504.19874] | ×~2 (part mémoire) selon régime |
| Block-GTQ (RoPE-aware quant) | 3.24× compression @128K, 1.34× decode | [2606.24033] | ×1.34 decode + ×3.24 mémoire |
| EAGLE-3 (spéculatif) | 3.0-6.5× harnais ; 2.36× réel SGLang bs=1 ; 1.32× batch 32+ | [2503.01840 Tables 1/4/3] | ×~1.3 en production chargée |
| vLLM/PagedAttention | 2-4× throughput | [2309.06180] | ×2-4 sur le coût mémoire |
| Sarathi-Serve | 2.6-5.6× capacité | [2403.02310] | ×2.6-5.6 sur le coût GPU amorti |
| DistServe | 7.4× req/s ou 12.6× SLO plus strict | [2401.09670] | ×7.4 goodput à SLO fixe |
| Mooncake (cache préfixe) | +525% throughput (Kimi) | [2407.00079] | ×6.25 throughput |
| TaiChi (switch agg/désagg) | +77% goodput | [2508.01989] | ×1.77 goodput |
| FlashInfer (kernels) | 29-69% ITL, +15.95% throughput | [2501.01005] | ×~1.16 throughput |
| DuetServe (SM multiplex) | +1.3× throughput | [2511.04791] | ×1.3 |
| SpargeAttention2 | 16.2× attention, 4.7× E2E | [2602.13515] | ×4.7 E2E |
| DashAttention | 3.36× vs FA-3 @96K | [2605.18753] | ×3.36 attention |
| NSA (entraînée) | 9.0× fwd, 11.6× decode attendu @64k | [2502.11089 Table 4] | ×11.6 decode |

DuetServe : arXiv 2511.04791 (vérifié sur le fichier corpus).

### 2.3 Estimation du coût marginal d'un token servi (modèle de coût)

[INFÉRENCE] En combinant DSA (coût réel) et les ratios V4/LSA, on peut estimer le coût marginal d'un token à 1M de contexte pour V4-Flash (base : decode V3.2 ≈ 0.8 $/M tokens à 128K [2512.02556]) :

- À 128K, V4-Flash ≈ 0.8 × (10%/37B-normalisé)... — prudence : les ratios V4 (10%) sont mesurés **à 1M**, pas à 128K, et le FLOPs/token croît avec L. Une extrapolation naïve est interdite. Ce qu'on peut dire :
  - À 1M tokens, le FLOPs/token de V4-Flash ≈ 1.1×10^9 vs V3.2 ≈ 13.2×10^9 (Fig. 1 gauche, décode) [FAIT] — soit ×12 ;
  - Le coût decode par token à 1M pour V4-Flash serait ~0.8/12 ≈ 0.07 $/M tokens **si le coût était purement proportionnel aux FLOPs** — ce qui est faux en pratique (le décodage est memory-bound : le coût dépend surtout de la lecture des poids et du KV) ;
  - Correction memory-bound : le décodage coûte ~ (poids actifs + KV lu)/Bande passante — V4-Flash lit 13B actifs (~26 GB FP8?) + KV(1M×~4%×...) — l'estimation exige le détail des poids FP4 et du KV compressé, non fourni par le corpus.

**Conclusion honnête** : le corpus ne permet PAS une conversion rigoureuse FLOPs→$/M tokens à 1M pour V4-Flash. Les seuls $/M tokens vérifiés sont ceux de V3.2 (0.2/0.8 à 128K) et la trajectoire V3.2→V4 est un facteur ~3-4× en FLOPs et ~10× en KV, pas un prix. [FAIT pour les limites]

### 2.4 Le point important : le coût mémoire domine en régime long-contexte

[INFÉRENCE] À 1M tokens, le KV cache d'une seule requête sous V3.2 (dense) serait ~70 Ko/token × 1M = ~70 Go — c'est-à-dire presque l'HBM d'un H800 (141 Go). Le modèle dense ne peut tout simplement pas servir le 1M. V4-Flash, à 7% de ce KV, le peut (≈ 4.9 Go par requête). Le vrai gain économique du natif n'est donc pas le FLOPs mais la **faisabilité mémoire** : servir du 1M tokens sans désagréger massivement le cache. La désagrégation (axe 4) devient alors l'outil pour multiplier la concurrence sur ce cache compressé.

### 2.5 Capital vs opérationnel

[FAIT] Entraînement V3 : 5.576 M$ ; V4 : non chiffré dans le corpus (32-33T tokens, ~14× celui de V3 en tokens... [2606.19348, §4.2.1] donne le planning mais pas un total $). Continued-pretraining DSA : 2 stages (1000 + 15000 steps, ~943.7B tokens) [2512.02556]. LSA indexer : ~1 GPU-hour H20 [2606.09079].
[INFÉRENCE] Le coût d'adoption d'une technique est dominé par le continued-pretraining (~1T tokens pour CSA/HCA d'après la trajectoire V4 : dense warm-up 1T + sparse phases) ; les surcouches (LSA) sont ~5 ordres de grandeur moins chères mais plafonnent plus bas (MRCR, 2× ceiling). C'est la structure coût/performance du domaine : chaque ×2 de gain exige soit un prétraining massif (architecture), soit une contrainte de tâche (surcouche).

---

## 3. Tensions économiques

1. **FLOPs vs mémoire vs bande passante** : les papiers optimisent des métriques différentes (FLOPs pour V4, KV GPU pour LSA, throughput pour Mooncake). Les convertir en $ exige un modèle de coût partagé, absent du corpus. Les chiffres $ réels de V3.2 ne couvrent qu'un point (128K, H800). [FAIT pour l'absence]
2. **Speedups bench vs production** : les speedups 3.0-6.5× d'EAGLE-3 (harnais batch 1, Table 1) coexistent avec ×2.36 (SGLang bs=1 réel, Table 4) et 1.32× (batch 32, Table 3) dans le même papier. Un décideur qui lit l'abstract paie un prix triple pour un gain tiers. [FAIT]
3. **Le coût caché de la désagrégation** : le transfert KV n'est pas gratuit (bande passante RDMA, cache Mooncake). DistServe ne chiffre pas le coût réseau dans ses $ ; Mooncake non plus (infrastructure existante). [INFÉRENCE]
4. **L'amortissement du prétraining** : les gains d'attention creuse exigent ~1T tokens de continued-pretraining — rentable uniquement pour un fournisseur modèle ; une équipe plateforme achète plutôt une surcouche (LSA, quantization) qui se déploie en jours. [INFÉRENCE]

---

## 4. Questions ouvertes (axe 5)

1. **Un modèle de coût partagé** : FLOPs, HBM-octets, réseau-octets, $/h — aucun papier du corpus n'en propose un qui couvre architecture+serving. Protocole : construire un estimateur paramétrique (comme le roofline mais avec prix) et l'étalonner sur les chiffres V3.2 publiés (0.2/0.8 $/M @128K) puis prédire le V4-Flash à 1M — la prédiction est falsifiable par les annonces de prix DeepSeek.
2. **Prix de vente vs coût** : les prix publics de l'API DeepSeek (hors corpus — à vérifier sur api-docs.deepseek.com) donnent la marge ; le corpus ne la couvre pas.
3. **Coût carbone/énergie** : aucun papier du corpus ne publie J/token. Lacune méthodologique générale du domaine (à ajouter au corpus : analyses energy-aware).
4. **Amortissement du prétraining** : à partir de quel volume de requêtes le continued-pretraining de CSA/HCA (~1T tokens) devient-il rentable vs une pile post-hoc (LSA+quant+serving) ? Dépend du prix du token — protocole de seuil de rentabilité.
