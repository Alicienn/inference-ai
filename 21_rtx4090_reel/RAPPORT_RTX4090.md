# Axe 21 — Mesures réelles sur GPU loué (RTX 4090, Qwen3-8B)

**Date :** 9 septembre 2026 · **Matériel :** RTX 4090 48 Go (vast.ai), CUDA 12.8 ·
**Modèle :** Qwen/Qwen3-8B réel (36 couches, 32 têtes Q / 8 têtes KV, head_dim 128) ·
**Coût :** < 3 $, session unique.

Premier test du projet sur un GPU loué avec un vrai modèle 8B (jusqu'ici : CPU seul,
petits modèles). Objectif : vérifier si les conclusions du papier (`17_papier/asp.tex`,
mesuré sur H200 loué) et du POC (`12_poc/`, mesuré sur petits modèles CPU) tiennent sur
une architecture différente et un modèle de taille de production.

---

## 1. Le gather irrégulier reste gratuit sur RTX 4090 (4ᵉ architecture)

Confirme `16_gpu/RAPPORT_GATHER.md` sur une 4ᵉ carte (après gfx1152/LPDDR5, T4/GDDR6,
A100/HBM2e) :

| morceau | contigu | gather | ratio |
|---:|---:|---:|---:|
| 512 o | 888,6 GB/s | 859,5 | 0,967 |
| 1024 o | 913,4 | 885,6 | 0,970 |
| 2048 o+ | 916-937 | 908-934 | 0,99-1,00 |

**[FAIT]** Pic mesuré : 936,5 GB/s. Pénalité de gather quasi nulle dès 512 octets — cohérent
avec la tendance déjà établie (la pénalité décroît avec le parallélisme du GPU).

**[RÉSERVE — trouvée après coup]** Le test complet du motif ASP à deux passes (32 configs,
volumes 4-16 Mio) donne une "efficacité brute" de 48,9 % avec 18/32 configs plus lentes
que le chemin plat. Une relecture indépendante a montré que **30 des 32 configurations du
chemin "plat" tenaient entièrement dans le cache L2 du GPU (72 Mio sur la 4090)** : le débit
implicite mesuré (822-3647 GB/s) dépasse le pic DRAM réel (933 GB/s), donc ce chemin ne
lisait pas la DRAM. **Ce tableau d'efficacité compare du cache contre de la DRAM, pas
ASP contre le plat en conditions réelles** — il ne doit pas être cité tel quel. Le seuil de
granularité (tableau ci-dessus, 96 Mio par lancement) n'est pas concerné, il dépasse déjà
la taille du L2. Un nouveau test à volumes systématiquement > 72 Mio serait nécessaire
pour un chiffre d'efficacité fiable. Voir `16_gpu/resultats/rtx4090_raw.txt`.

---

## 2. Bout en bout, Qwen3-8B réel : ASP est plus lent, comme sur H200

**[FAIT]** Décodage complet (pas seulement l'attention), poids réels chargés, cache KV réel :

| contexte | KV cache | dense | ASP | gain vitesse | octets économisés |
|---:|---:|---:|---:|---:|---:|
| 16 384 | 2,2 Go | 113,95 ms | 212,94 ms | **0,54×** | 13,5× |
| 32 768 | 4,5 Go | 113,55 ms | 280,34 ms | **0,41×** | 17,7× |
| 65 536 | 9,0 Go | 188,44 ms | 212,14 ms | **0,89×** | 20,9× |
| 131 072 | 18,0 Go | 198,99 ms | 219,19 ms | **0,91×** | 23,0× |
| 262 144 / 524 288 | 36 / 72 Go | — | — | — | ignoré (VRAM insuffisante, 48 Go) |

**Confirme, sur un 2ᵉ GPU indépendant (RTX 4090 après H200), la conclusion centrale du
papier** : ASP lit 13 à 23× moins d'octets mais est 10 à 145 % **plus lent** en pratique.
Code : `16_gpu/e2e_qwen.py`. Résultats : `16_gpu/resultats/e2e_qwen_rtx4090.json`.

## 3. Ventilation : où passe le temps

**[FAIT]** Bande passante RTX4090 mesurée injectée (`ASP_BW=936.5e9`), attention seule
(hors reste du modèle) :

| N | KV | plancher mémoire | attention dense | attention ASP |
|---:|---:|---:|---:|---:|
| 16 384 | 2,2 Go | 20,12 ms | 2,59 ms | 25,37 ms |
| 65 536 | 9,0 Go | 27,86 ms | 12,01 ms | 22,39 ms |
| 131 072 | 18,0 Go | 38,18 ms | 22,33 ms | 25,30 ms |
| 262 144 | 36,0 Go | 58,82 ms | 42,66 ms | 24,15 ms |

**[INFÉRENCE]** Le surcoût ASP est quasi constant (~22-25 ms) alors que l'attention dense
croît avec le contexte — la bascule se situe entre 131k et 262k tokens sur RTX 4090,
plus tôt que sur H200 (papier : ~470k). Cohérent avec le mécanisme identifié dans le
papier (surcoût de framework non fusionné), pas avec une limite propre à ASP.
Résultats : `16_gpu/resultats/vent_rtx4090.json`.

---

## 4. Qualité du sélecteur sur le vrai 8B : coreset confirme, bat COBS

**[FAIT]** Capture Q/K réelle sur Qwen3-8B (8192 tokens, `12_poc/code/capture_qk.py`
adapté GPU/bf16). Sélection de blocs distants, L=64, budget égal :

| méthode | coût (octets/D) | rappel @8 (RoPE) | rappel @8 (sans RoPE) |
|---|---:|---:|---:|
| coreset r=2 | 2,02 | 82,70 % | 83,69 % |
| cobs r=2 | 3,02 | 77,42 % | 77,23 % |
| coreset r=4 | 4,03 | 86,64 % | 87,31 % |
| cobs r=4 | 5,03 | 79,87 % | 79,45 % |
| **coreset r=8** | **8,06** | **89,88 %** | **91,31 %** |
| cobs r=8 | 9,06 | 81,48 % | 80,72 % |

**Confirme sur un vrai modèle 8B de production** ce que le POC avait trouvé sur des petits
modèles CPU (Qwen2.5-0.5B, SmolLM2-135M) : coreset domine COBS à coût égal ou inférieur.
Résultats : `12_poc/resultats/selection_distant_L64.json`, log complet
`12_poc/resultats/eval_qwen8b_gpu.log`.

## 5. Loi octets-précision sur le vrai 8B

**[FAIT]** `19_loi_octets/code/frontiere2.py` sur la capture Qwen3-8B réelle :

| octets | méthode | rappel @8 |
|---:|---|---:|
| 258 | mean / coreset r=1 | 73,0 % |
| 516 | coreset r=2 | 83,2 % |
| 2050 | top r=8 (clés de plus grande norme) | 92,6 % |
| 2560 | quant 2b | 97,9 % |
| 4608 | quant 4b | 99,9 % |

**[FAIT]** La quantification reste, de loin, le meilleur rapport octets/précision — confirmé
sur le vrai 8B (facteur ~8-9× par bit gagné, cf. `19_loi_octets/RAPPORT_OCTETS.md`).

**[NOUVEAU, à creuser]** La famille "top" (garder les r clés de plus grande norme, sans
aucun calcul de centroïde) bat systématiquement coreset et cobs sur Qwen3-8B — inattendu,
absent des petits modèles testés jusqu'ici. Hypothèse non vérifiée : tête_dim=128 (2× les
petits modèles) change la géométrie de norme des clés. **[HYPOTHÈSE]** À vérifier avant
de généraliser. Résultats : `19_loi_octets/resultats/frontiere_qwen8b_gpu.json`.

---

## 6. Pourquoi le sélecteur est-il lent ? Trois pistes testées, un vrai coupable trouvé

Un agent de recherche externe a proposé plusieurs leviers pour abaisser le seuil de
croisement (le contexte à partir duquel ASP devient plus rapide que dense, estimé à
N*≈146k sur RTX4090 à partir d'un modèle linéaire sur les données de la section 3). On a
testé les trois idées les moins coûteuses.

### Test A — réutiliser la sélection sur plusieurs pas de décodage : réfutée

**[FAIT]** Sur la capture réelle Qwen3-8B (gratuit, sans GPU), chevauchement (indice de
Jaccard) du top-k de blocs sélectionnés entre le pas p et le pas p+écart :

| écart (pas) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| chevauchement | 0,509 | 0,434 | 0,389 | 0,371 | 0,360 | 0,337 |

**[FAIT]** Même d'un pas au suivant, à peine la moitié des blocs sélectionnés sont
identiques. Réutiliser la sélection sur plusieurs pas perdrait trop de masse d'attention :
idée écartée. Code : `21_rtx4090_reel/test_a_stabilite_selection.py`.

### Test B — décomposition du "plateau" : ce n'est pas un coût fixe, c'est un bug d'archi

**[FAIT]** Décomposition du coût du sélecteur par couche en 6 étapes (RTX4090) :

| N | construction résumés | tri+gather+attention (reste) |
|---:|---:|---:|
| 16 384 | 0,054 ms | 0,541 ms |
| 65 536 | 0,339 ms | 0,536 ms |
| 131 072 | 0,641 ms | 0,521 ms |
| 262 144 | **1,251 ms** | 0,541 ms |

**[FAIT]** Seule la construction des résumés de blocs croît avec le contexte (×23 entre
16k et 262k) ; le reste est plat. **[INFÉRENCE]** Le modèle "plateau fixe ~24ms" de la
section 3 sous-estime donc le coût réel : ce plateau ne provenait à l'origine que d'un
artefact de mesure (`vent.py` réutilisait les résumés déjà construits d'un appel à
l'autre, ce qu'un vrai décodage ne fait jamais — le contexte change de token à chaque
pas). En production, l'implémentation actuelle **reconstruit tous les résumés de blocs à
partir de zéro à chaque mot généré**, alors qu'une mise à jour incrémentale du seul
dernier bloc modifié suffirait. C'est un défaut d'implémentation plus simple à corriger
qu'une fusion complète de noyau (qui reste utile mais cible le mauvais composant en
premier). Code : `16_gpu/probe_b_decomposition.py`.

### Test C — partage du coût entre requêtes d'un batch : non concluant

**[FAIT]** À N=65 536 fixe (VRAM limitée à B=2 sur 48 Go à ce contexte) :

| batch | dense / séquence | ASP / séquence | ratio |
|---:|---:|---:|---:|
| 1 | 55,29 ms | 64,72 ms | 0,854× |
| 2 | 44,31 ms | 47,95 ms | 0,924× |

**[LIMITE]** Seulement 2 points (VRAM insuffisante au-delà). Le gain existe mais profite
presque autant au dense qu'à ASP — pas l'effet "quasi gratuit" espéré, mais l'échantillon
est trop petit pour trancher. Code : `16_gpu/probe_c_batch.py`.

---

## 7. Ce que ça change / ne change pas

- **Ne change pas** : la conclusion centrale du papier (ASP économise des octets mais ne
  devient pas plus rapide sans fusion de noyau) — **renforcée**, reproduite sur un 2ᵉ GPU.
- **Ne change pas** : aucun impact sur l'entraînement — ce travail est 100 % inférence
  (lecture du cache KV en décodage).
- **Renforce** : le résultat du POC (coreset > COBS) généralise du CPU/petits modèles au
  GPU/8B réel.
- **Nouvelle réserve** : les chiffres d'efficacité GPU du micro-test isolé
  `16_gpu/RAPPORT_GATHER.md` (motif ASP complet, pas le test de granularité) sont
  partiellement contaminés par le cache L2 et ne doivent pas être cités sans cette réserve.
- **Piste ouverte** : l'intérêt réel d'économiser des octets n'est probablement pas la
  latence par requête (négative ici) mais le débit multi-utilisateurs (plus de séquences
  servies en parallèle à mémoire GPU égale) — non mesuré dans cette session.
- **Nouveau, le plus actionnable** : le vrai levier pour améliorer la latence n'est pas la
  fusion de noyau (ciblait le mauvais composant) mais la **mise à jour incrémentale des
  résumés de blocs**, actuellement reconstruits en entier à chaque token généré (§6,
  Test B). Prochaine étape naturelle si quelqu'un poursuit ce travail.

## Reproductibilité

- `16_gpu/bench_gather.py` / `colab_driver.py` (gather), `e2e_qwen.py` (bout en bout),
  `vent.py` (ventilation, patché pour bande passante configurable via `ASP_BW`).
- `12_poc/code/capture_qk.py` (patché GPU bf16 + fallback dataset `Salesforce/wikitext`),
  `eval_selection.py --tag qwen8b_gpu`.
- `19_loi_octets/code/frontiere2.py --tags qwen8b_gpu` (patché chemins portables).
- Dépôt complet (code, sans les gros binaires) : github.com/Alicienn/inference-ai.
