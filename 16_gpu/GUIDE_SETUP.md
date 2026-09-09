# Voies de confirmation — ROCm/HIP et GPU cloud

> **Statut : optionnel.** La question a déjà été tranchée sur votre GPU réel (`gfx1152`)
> via OpenCL, sans aucune installation. Voir [`RAPPORT_GATHER.md`](RAPPORT_GATHER.md).
> Ce qui suit sert à **confirmer sur une autre pile logicielle** (HIP) ou sur une autre
> **architecture mémoire** (NVIDIA/HBM), ce qui est la vraie limite du résultat actuel.

---

## A. Ce qui a été vérifié sur votre machine

| Élément | Constat |
|---|---|
| GPU | AMD Radeon 860M, pilote `32.0.31041.1004` |
| CPU | Ryzen AI 7 350, 8 cœurs / 16 threads |
| ROCm / HIP | **absent** (`hipcc`, `rocminfo`, `hipconfig`, `rocm-smi` introuvables) |
| OpenCL | **présent et fonctionnel** — device exposé sous le nom `gfx1152` |
| WSL2 | installé, distribution par défaut `kali-linux`, version 2 |

C'est la présence d'OpenCL qui a permis de contourner le blocage : le pilote Adrenalin
expose l'iGPU en OpenCL 2.1, ce qui suffit à mesurer des motifs d'accès mémoire.

---

## B. Piste 1 — ROCm/HIP (confirmation sur la pile officielle)

### B.1 Point de vigilance sur les versions

D'après la documentation AMD, c'est **ROCm 10.0.0** qui liste officiellement le
Ryzen AI 7 350 / Radeon 860M (`gfx1152`) — et **sous Linux**. Il n'existe pas, à ma
lecture, de build `gfx1152` pour le **HIP SDK Windows**
([issue ROCm/TheRock #1980](https://github.com/ROCm/TheRock/issues/1980),
[matrice de compatibilité](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)).
Votre référence à « ROCm 7.13, mai 2026 » ne correspond pas à ce que j'ai pu vérifier —
**à confirmer de votre côté**, je peux me tromper sur la numérotation.

**Conséquence pratique : viser Linux (WSL2 ou natif), pas le HIP SDK Windows.**

### B.2 Checklist vérifiable, étape par étape

Rapportez-moi la sortie de chaque étape ; je débogue à partir de là.

**Étape 1 — une distribution WSL2 Ubuntu (Kali n'est pas supportée par ROCm)**
```bash
wsl --install -d Ubuntu-24.04
```

**Étape 2 — dans le WSL Ubuntu, vérifier que le GPU est visible**
```bash
ls -l /dev/dxg && echo "OK: passthrough GPU WSL present"
```
Si `/dev/dxg` est absent : mettre à jour WSL (`wsl --update`) et le pilote Adrenalin.

**Étape 3 — installer le dépôt ROCm**
```bash
sudo apt update && sudo apt install -y python3-setuptools python3-wheel wget
```

**Étape 4 — installer ROCm (adapter le numéro de version à celui que vous visez)**
```bash
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
```

**Étape 5 — vérifier que le GPU est reconnu et son nom d'architecture**
```bash
rocminfo | grep -i gfx
```
Attendu : `gfx1152`. Si `rocminfo` ne voit rien, ROCm-on-WSL ne supporte pas ce chip
et il faut passer à Linux natif (dual boot ou live USB).

**Étape 6 — contournement si `gfx1152` n'est pas reconnu**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```
(usurpation d'identité vers une architecture proche ; fonctionne souvent pour les iGPU
RDNA 3.5, à vos risques)

**Étape 7 — compiler et lancer le banc**
```bash
hipcc -O3 --offload-arch=gfx1152 asp_bench.cpp -o asp_bench && ./asp_bench
```

### B.3 Sur le portage CUDA → HIP

[`asp_bench.cpp`](asp_bench.cpp) est une **source unique** compilable par `nvcc` et par
`hipcc`. Les points fragiles de la traduction automatique (`hipify-perl`) ont été
traités à la main :

- **Largeur de wavefront** — 32 sur NVIDIA, 32 ou 64 sur RDNA selon le mode, 64 sur
  CDNA. Le noyau n'utilise **aucune primitive de warp** (`__shfl_*`, `__ballot`) : la
  réduction passe par la mémoire partagée avec `__syncthreads()`. Le code est donc
  **insensible à la largeur du wavefront** — c'est le choix qui rend le portage sûr.
- **Mémoire partagée** — 48 Kio/bloc sur NVIDIA, 64 Kio de LDS sur RDNA. On reste à
  `256 × 16 = 4 Kio`, très en dessous des deux plafonds.
- **Conflits de banques** — la réduction accède `red[lid]` et `red[lid+s]`, motif sans
  conflit sur les deux architectures (banques de 4 octets, accès `uint4` alignés).
- **`uint4`** existe des deux côtés avec la même sémantique.

---

## C. Piste 2 — GPU NVIDIA cloud (la confirmation qui compte vraiment)

C'est la piste **la plus utile scientifiquement** : elle teste une architecture mémoire
radicalement différente (GDDR6/HBM, 4 à 40× plus de bande passante) là où le 860M est en
LPDDR5 partagée à ~80 Go/s.

### ⚡ Voie retenue : `google-colab-cli` depuis WSL2 — DÉJÀ INSTALLÉ

Le CLI permet de louer un GPU, exécuter un script et libérer la machine, sans navigateur
et sans notebook. Il est **Linux/macOS uniquement**, d'où l'usage de WSL2.

État de l'installation sur cette machine :

| Élément | État |
|---|---|
| WSL2 | ✅ `kali-linux` (v2), Python 3.13, `uv` présent |
| `google-colab-cli` | ✅ installé via `uv tool install`, binaire `~/.local/bin/colab` |
| [`colab_driver.py`](colab_driver.py) | ✅ pilote autonome (source CUDA embarquée, compile, exécute) |
| Authentification | ⏳ **une action manuelle requise** (flux OAuth navigateur) |

**L'unique étape manuelle.** Dans un terminal WSL :

```bash
wsl -e bash -lc "cd /mnt/c/Users/alici/Downloads/inference_opti/16_gpu && colab sessions"
```

Le CLI affiche une URL Google, à ouvrir dans un navigateur ; après approbation Google
affiche un code d'autorisation, à recoller dans le terminal. Les identifiants sont
ensuite mis en cache et le CLI devient utilisable sans interaction.

**Ensuite, le banc tourne en une commande** (~2 min, T4 gratuit) :

```bash
wsl -e bash -lc "cd /mnt/c/Users/alici/Downloads/inference_opti/16_gpu && colab run --gpu T4 colab_driver.py"
```

### ⚠️ Le TPU v5e-1 ne convient pas pour CE test

`asp_bench.cpp` est du CUDA/HIP : un TPU n'a ni CUDA, ni warps, ni la notion de
coalescence qu'on cherche justement à mesurer. Il faut un runtime **GPU**. Le T4 (gratuit)
suffit pour trancher. Un **A100** (HBM2e) serait le test idéal du modèle de seuil, s'il
est accessible avec l'abonnement — c'est la seule mémoire HBM directement testable ici.

### Accélérateurs disponibles via le CLI

GPU : `T4`, `L4`, `G4`, `H100`, `A100` — TPU : `v5e1`, `v6e1` (inutiles ici).
La disponibilité dépend de l'abonnement ; T4 est sur l'offre gratuite.

### Prédictions posées AVANT la mesure — comment lire le résultat

Le modèle MLP (loi de Little, [`threshold_model.py`](threshold_model.py)) prédit
`C* = min(6144, 3,28·B·λ/N_res)` :

| GPU | seuil C* prédit | lecture |
|---|---:|---|
| gfx1152 (mesuré) | 1024 o | référence |
| **T4** | **328 o** | **plus bas** que l'iGPU : le gather y serait encore plus bénin |
| A100 | 1009 o | identique |
| H100 | 1559 o | la règle passerait à ~2 Kio |
| MI300X | 2321 o | la règle passerait à ~2,5 Kio |

**Le modèle est réfuté** si le seuil mesuré sur T4 est *plus haut* que 1024 o : cela
voudrait dire que le phénomène n'est pas le parallélisme mémoire mais la structure DRAM,
et il faudrait remplacer `B·λ/N_res` par un terme en taille de ligne DRAM.
Le paramètre le plus incertain est λ : s'il est 2× plus grand que supposé sur HBM3, le
seuil H100 doublerait à ~3,1 Kio.

### Variante sans CLI : le notebook [`ASP_Colab.ipynb`](ASP_Colab.ipynb) (importer, GPU T4, tout exécuter)

### Variante entièrement manuelle

1. Ouvrir [colab.research.google.com](https://colab.research.google.com), nouveau notebook.
2. Menu **Exécution → Modifier le type d'exécution → GPU T4**.
3. Coller ceci dans une cellule et exécuter :

```python
!nvidia-smi --query-gpu=name,memory.total,clocks.max.memory --format=csv
```

4. Coller le contenu de [`asp_bench.cpp`](asp_bench.cpp) dans une cellule préfixée
   `%%writefile asp_bench.cpp`, puis dans une cellule suivante :

```bash
!nvcc -O3 -arch=sm_75 asp_bench.cpp -o asp_bench && ./asp_bench
```

(`sm_75` pour T4 ; `sm_80` pour A100, `sm_89` pour L4, `sm_90` pour H100.)

5. Me rapporter la table de sortie. La colonne qui tranche est **`efficacite`** :
   si elle reste au-dessus de ~85 % sur HBM, le résultat se généralise ; si elle
   s'effondre, c'est que le mécanisme dépend de la hiérarchie mémoire et le résultat
   du 860M ne se transpose pas.

**Alternatives** si Colab n'est pas disponible : Kaggle Notebooks (P100/T4 gratuits,
30 h/semaine), Lightning AI Studio (crédits gratuits mensuels), ou une location à
l'heure (Vast.ai, RunPod — une RTX 4090 coûte typiquement moins de 0,50 $/h et le
banc tourne en moins d'une minute).

---

## D. Piste 3 — proxy CPU

[`proxy_cpu.py`](proxy_cpu.py), exécuté, résultats dans
[`resultats/proxy_cpu.json`](resultats/proxy_cpu.json). **[SIGNAL FAIBLE — non
transposable au GPU]** : il mesure la sensibilité d'une hiérarchie de caches CPU à un
accès dispersé, pas la coalescence de wavefront. Il ne sert que de contre-point et
n'entre dans aucune conclusion.
