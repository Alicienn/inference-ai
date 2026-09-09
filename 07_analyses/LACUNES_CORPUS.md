# Lacunes du corpus — ce qui manquait, ce qui a été ajouté, ce qui manque encore

Le corpus initial (38 papiers) couvrait bien l'attention creuse, le KV cache, la spéculation et le
serving. L'analyse a révélé des angles morts. **18 papiers ont été ajoutés** pour les combler
(corpus porté à 49). Cette note explique pourquoi chacun était nécessaire, et ce qui reste absent.

---

## 1. Lacunes comblées

### 1.1 Quantification des **poids** — dossier `08_quantization_poids/` (4 papiers)

**Pourquoi c'était nécessaire.** Le corpus initial ne traitait que la quantification du *KV cache*.
Or le résultat de l'axe 4 le plus lourd de conséquences est que **V4-Flash tient sur un seul B200
grâce au FP4 sur les experts routés** (135 GB au lieu de 270). C'est de la quantification de poids,
et il n'y avait rien dessus.

| Papier | Apport |
|---|---|
| AWQ ([2306.00978](https://arxiv.org/abs/2306.00978)) | Quantification consciente des activations ; référence des canaux saillants |
| GPTQ ([2210.17323](https://arxiv.org/abs/2210.17323)) | Post-training quantization de référence |
| QuaRot ([2404.00456](https://arxiv.org/abs/2404.00456)) | Inférence 4 bits sans outliers par **rotation** |
| SpinQuant ([2405.16406](https://arxiv.org/abs/2405.16406)) | Rotations **apprises** |

**[INFÉRENCE]** QuaRot et SpinQuant éclairent aussi l'axe 2 : ils emploient le même levier que
TurboQuant (rotation pour égaliser l'énergie et détruire les outliers), qui est **en tension
directe** avec Block-GTQ, lequel *exploite* l'inégalité d'énergie. Avoir les deux familles dans le
corpus permet d'instruire cette tension (question Q4).

### 1.2 Inférence MoE — dossier `09_moe_inference/` (3 papiers)

**Pourquoi.** V4-Flash est un MoE extrême (13 B actifs sur 284 B, 6 experts sur 256). Le corpus
initial ne contenait **aucun** papier sur l'inférence MoE : ni offloading d'experts, ni caching, ni
affinité inter-couches. Or c'est là que se joue la lecture des poids en décodage — le terme que
mon modèle roofline montre dominant à faible batch.

- Fast Inference of MoE with Offloading ([2312.17238](https://arxiv.org/abs/2312.17238))
- Towards MoE Deployment ([2303.06182](https://arxiv.org/abs/2303.06182))
- Inter-Layer Expert Affinity ([2401.08383](https://arxiv.org/abs/2401.08383))

### 1.3 Cache de préfixes et attention sink — ajouts ciblés

**Pourquoi.** L'axe 4 conclut que le régime décisif de la désagrégation est la **réutilisation de
préfixe** — or le papier de référence sur le sujet (SGLang / RadixAttention) était absent. De même,
V4 emploie l'*attention sink* (§2.3.3) sans que le papier fondateur soit au corpus.

- SGLang / RadixAttention ([2312.07104](https://arxiv.org/abs/2312.07104)) → `05_serving_systems/`
- StreamingLLM / attention sink ([2309.17453](https://arxiv.org/abs/2309.17453)) → `03_attention_kv_cache/`
- Infini-attention ([2404.07143](https://arxiv.org/abs/2404.07143)) → `03_attention_kv_cache/` — mémoire compressive, alternative à CSA/HCA
- ARKV ([2603.08727](https://arxiv.org/abs/2603.08727)) → budget KV adaptatif
- Load-Aware Prefill Deflection ([2607.02043](https://arxiv.org/abs/2607.02043)) → `05_serving_systems/`

### 1.4 Sécurité de la spéculation

**Pourquoi.** L'axe 3 montre que le décodage spéculatif est fragile (l'accélération peut tomber à
1,00×). Il existe une littérature d'**attaques** qui exploite précisément cette fragilité, absente
du corpus.

✅ RÉSOLU (ajouté & vérifié) — - Mistletoe ([2605.14005](https://arxiv.org/abs/2605.14005)) → attaques par effondrement
  d'accélération sur le décodage spéculatif. Directement pertinent en déploiement partagé.

### 1.5 Évaluation et métriques — dossier `10_benchmarks_eval/` (2 papiers)

**Pourquoi.** Toutes mes mesures utilisent des proxys (masse d'attention, KL). Le corpus ne
contenait aucun benchmark de contexte long ni aucune réflexion sur les métriques de coût.

- RULER ([2404.06654](https://arxiv.org/abs/2404.06654)) — « quelle est la vraie taille de contexte
  de votre modèle ? » Indispensable pour valider les affirmations 1M de V4.
- Energy-to-Token ([2605.11733](https://arxiv.org/abs/2605.11733)) — position : évaluer
  l'inférence en énergie par token. Complète ma métrique « octets lus par token ».

### 1.6 Matériel — dossier `11_materiel/` (2 papiers)

**Pourquoi.** Mon analyse roofline conclut que le décodage est borné par la bande passante et que
le point de bascule ne s'améliore pas d'une génération à l'autre. Les pistes matérielles qui
changeraient cette donne n'étaient pas représentées.

- 3D-DRAM pour accélérateurs LLM ([2604.08044](https://arxiv.org/abs/2604.08044))
- SMEPilot / Scalable Matrix Extensions ([2606.16332](https://arxiv.org/abs/2606.16332)) — inférence CPU

### 1.7 Serving des modèles de raisonnement

- Étude empirique du serving des *reasoning models* ([2510.18672](https://arxiv.org/abs/2510.18672))
  → `02_surveys/`. Les modèles à longue chaîne de pensée ont un profil décode/prefill très
  différent, non couvert par les papiers de serving classiques.

---

## 2. Lacunes qui subsistent

Classées par gravité pour la suite du travail.

### 2.1 ⛔ Aucun papier sur la quantification d'un **cache latent partagé** (MLA / CSA)

C'est l'angle mort le plus sérieux, et il est **structurel, pas éditorial** : à ma connaissance
cette littérature n'existe pas encore. Toute la quantification KV du corpus suppose des K et V
distincts (GQA). MLA et CSA stockent un état latent unique servant des deux rôles, avec deux
chemins de sensibilité différents. → **Question Q3**, potentiellement un papier à écrire.

### 2.2 ⛔ Pas d'implémentation de référence de CSA/HCA

Le papier V4 renvoie à `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference` (note 1).
Ce code n'est pas dans le corpus et lèverait deux hypothèses sensibles de mes calculs : la
précision du cache d'indexation (FP4 supposé) et la répartition exacte CSA/HCA par couche.
**Priorité de récupération n°1** (voir [`METHODES.md`](METHODES.md) §7).

### 2.3 ⚠️ Peu de choses sur le batching continu et le scheduling sous contrainte de SLO

Sarathi-Serve et FlowPrefill couvrent le prefill. La théorie du batching continu côté décode
(Orca, et les travaux de scheduling sous SLO) reste peu représentée.

### 2.4 ⚠️ Rien sur l'inférence agentique / multi-tours

V4-Flash est explicitement positionné pour « l'inférence agentique à bas coût » et V4 intègre des
données agentiques en mid-training. Or la charge agentique (préfixes très réutilisés, sorties
courtes, nombreux allers-retours) a un profil de coût très différent — c'est précisément le régime
où mon axe 4 montre que la réutilisation de préfixe devient décisive. Aucun papier ne le
caractérise quantitativement.

### 2.5 ⚠️ Peu de données de reproduction indépendante

Presque tous les chiffres du corpus sont auto-rapportés. Un seul papier de mesure tierce
(2510.18672). Pour un domaine dont les affirmations sont chiffrées, c'est fragile — d'où ma
démarche de reconstruction indépendante (SYNTHESE §2).

### 2.6 ℹ️ Absences assumées

- Distillation et compression de modèle (hors périmètre : c'est de l'entraînement).
- Inférence sur périphérique / edge, sauf via SMEPilot.
- Accélérateurs non-NVIDIA, sauf EAGLE-Pangu (Ascend) et SMEPilot (CPU).

---

## 3. État final du corpus

| Dossier | Papiers | Thème |
|---|---:|---|
| `01_deepseek/` | 6 | Lignée DeepSeek V3 → V4 |
| `02_surveys/` | 6 | États de l'art |
| `03_attention_kv_cache/` | 10 | Attention efficace, compression KV |
| `04_speculative_decoding/` | 5 | Décodage spéculatif (+ sécurité) |
| `05_serving_systems/` | 11 | Serving, désagrégation, cache de préfixes |
| `08_quantization_poids/` | 4 | Quantification des poids |
| `09_moe_inference/` | 3 | Inférence MoE, offloading d'experts |
| `10_benchmarks_eval/` | 2 | Benchmarks et métriques |
| `11_materiel/` | 2 | Pistes matérielles |
| **Total** | **49** | |
- ✅ FlashInfer [arXiv 2501.01005] — PDF ajouté au corpus (05_serving_systems), texte extrait.
