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

### Test D (idée propre) — corriger le bug sur une vraie boucle de décodage : non concluant

**[FAIT]** Implémentation d'une mise à jour incrémentale des résumés (seul le dernier bloc,
partiel, est recalculé à chaque pas — O(Lb) au lieu de O(N)), testée sur une vraie boucle de
64 pas consécutifs où le cache grandit token par token (aucun test précédent, y compris
`e2e_qwen.py` et `vent.py`, ne mesurait plus d'un seul pas à N fixe) :

| N0 | ASP actuel / dense | ASP "corrigé" / dense |
|---:|---:|---:|
| 16 384 | 0,945× | 0,941× |
| 65 536 | 1,008× | 0,752× |
| 131 072 | 1,674× | **2,183×** (pire) |
| 262 144 | 1,022× | 0,751× |

**[LIMITE]** Pas de tendance cohérente — la version "corrigée" gagne parfois, perd parfois.
**[INFÉRENCE]** L'implémentation de test elle-même est probablement en cause, pas
l'hypothèse : la mise à jour incrémentale utilise `torch.cat` pour faire grandir les
tenseurs de résumés à chaque pas, ce qui réalloue et recopie de la mémoire à chaque appel
— un coût qui peut dominer le gain recherché à cette échelle. Démontrer proprement le
bénéfice demanderait des tampons pré-alloués (pas de `cat` en boucle), hors du périmètre
d'un script de test rapide. **Le diagnostic du Test B reste valide** (le recalcul complet
est le vrai coût qui grossit) ; ce Test D ne l'infirme pas, il échoue seulement à
démontrer proprement le correctif. Code : `16_gpu/probe_d_incremental.py`.

---

## 7. Le correctif implémenté proprement : ça marche

Suite au Test D (raté à cause de `torch.cat`), implémentation propre d'ASP v2
(`16_gpu/e2e_qwen_v2.py`) avec les deux corrections validées cette session :
- **Tampons à croissance amortie** (doublement de capacité, jamais de recopie à chaque
  pas) au lieu de reconstruire tout à chaque appel.
- **Score par MAX aux deux étages** (index PCA causal d′=8, base figée sur 256 tokens,
  pour le filtrage grossier ; max sur clés brutes pour les survivants) au lieu de la
  moyenne, confirmée catastrophique (§ vérification `test_e_mean_vs_max_premier_etage.py`).

**[FAIT]** Instantané à N fixe (même méthodologie que `e2e_qwen.py`, donc les deux variantes
bénéficient également de la mise en cache après amorçage — ne teste que l'effet de
l'algorithme de score, pas encore le gain incrémental) :

| N | dense | ASP actuel (bug) | ASP corrigé | gain ASP actuel | gain ASP corrigé |
|---:|---:|---:|---:|---:|---:|
| 16 384 | 40,74 ms | 67,94 ms | 60,93 ms | 0,60× | 0,67× |
| 32 768 | 39,55 ms | 65,65 ms | 60,78 ms | 0,60× | 0,65× |
| 65 536 | 54,54 ms | 65,52 ms | 60,67 ms | 0,83× | 0,90× |

Le correctif est systématiquement 5 à 12 % plus rapide que le bug d'origine, mais reste
plus lent que dense dans ce protocole à instantané unique. Résultats :
`16_gpu/resultats/e2e_qwen_v2.json`.

**[FAIT] Sur une vraie boucle de décodage (Test F, 128 pas consécutifs, cache qui grandit
de 1 token/pas)** — reprise du Test D avec les tampons corrects, isolant l'effet de
l'incrémentalité seule :

| N0 | ASP actuel (bug, O(N)/pas) | ASP corrigé (O(1) amorti/pas) | gain |
|---:|---:|---:|---:|
| 16 384 | 0,0136 s | 0,0321 s | 0,42× (pire) |
| 65 536 | 0,0811 s | 0,0661 s | **1,23×** |
| 131 072 | 0,1614 s | 0,1080 s | **1,49×** |
| 262 144 | 0,3210 s | 0,1912 s | **1,68×** |

**Le gain croît avec le contexte, exactement comme prédit par le diagnostic du Test B**
(coût O(N)/pas vs O(1) amorti/pas — l'écart absolu s'agrandit avec N). En dessous de
~65k tokens, le coût fixe de la base PCA domine et le correctif perd ; au-dessus, il
gagne de plus en plus. **C'est la première mesure de vitesse propre et cohérente de ce
fil de correctifs** — contrairement au Test D, le signal est net et dans le sens attendu.
Code : `16_gpu/e2e_qwen_v2.py`, `16_gpu/probe_f_multistep_v2.py`. Résultats :
`16_gpu/resultats/probe_f_multistep_v2.json`.

**[LIMITE]** Ce test isole la construction des résumés seule (comme le Test B/D), pas le
pipeline complet (tri + gather + attention finale) ni un vrai passage de modèle bout en
bout sur une séquence multi-pas — ce dernier reste à faire pour confirmer que le gain
survit une fois intégré dans un vrai passage `forward()` répété.

## 8. Le test décisif : intégré dans un vrai décodage complet, le correctif régresse

Dernière session GPU. Intégration du pipeline ASP2 complet (déjà fonctionnel : tampons
incrémentaux + score max aux deux étages) dans une **vraie boucle `model()` avec
`use_cache=True`**, le cache grandissant naturellement token par token — le test qui
manquait, ni un instantané figé ni une brique isolée.

**[FAIT]** 32 pas de décodage réels, Qwen3-8B complet (`16_gpu/e2e_multistep_final.py`) :

| N0 | dense | asp1 (bug d'origine) | asp2 (corrigé) | gain asp1 | gain asp2 |
|---:|---:|---:|---:|---:|---:|
| 16 384 | 1,68 s | 2,25 s | 2,86 s | 0,75× | **0,59×** |
| 65 536 | 1,74 s | 2,06 s | 2,77 s | 0,85× | **0,63×** |

**Le pipeline complet corrigé est plus lent que le bug d'origine, pas plus rapide** —
l'inverse de ce que le Test F isolé laissait espérer. VRAM insuffisante pour tester
131k/262k dans ce protocole (cache grandissant + poids + tampons simultanés).

**[FAIT] Diagnostic (Test G, `16_gpu/probe_g_decompose_asp2.py`)** — décomposition du coût
d'ASP2 par étape, à comparer à celle d'ASP1 (§6, Test B) :

| N | update (résumés) | tri grossier (stage1) | gather | tri fin (stage2) | gather final | attention |
|---:|---:|---:|---:|---:|---:|---:|
| 16 384 | 0,009 ms | 0,189 ms | 0,063 ms | 0,143 ms | 0,162 ms | 0,045 ms |
| 262 144 | 0,007 ms | **0,489 ms** | 0,207 ms | 0,139 ms | 0,159 ms | 0,053 ms |

**[FAIT]** `update` (la brique corrigée) est bien redevenu quasi gratuit (0,007-0,009 ms,
conforme au Test F) — **mais le coût s'est simplement déplacé** vers le tri grossier
(stage1), qui grossit maintenant lui-même avec le contexte (×2,6 entre 16k et 262k) :
scorer chaque bloc par un max sur ses clés projetées touche un tenseur à 5 dimensions
(blocs × clés du bloc × dimensions projetées) à chaque pas, même une fois les résumés
construits — alors qu'ASP1 réduisait ce même travail en une seule passe `logsumexp`
compacte. **[INFÉRENCE]** La somme des étapes d'ASP2 (0,613-1,054 ms/couche) reste
pourtant *inférieure* à celle d'ASP1 à grand contexte (1,792 ms/couche à 262k, Test B) —
l'écart mesuré dans le vrai modèle (§8) est donc probablement dominé par un surcoût
d'intégration non capturé par la mesure isolée (plus d'opérations GPU distinctes par
couche pour le score max — 8 environ contre 6 pour ASP1 — donc plus de lancements de
calcul séparés, dont le coût s'additionne différemment une fois entrelacés avec les 36
couches du vrai modèle). Non vérifié plus avant, faute de budget GPU restant.

**Conclusion honnête de cette piste de correctif** : le bug de vitesse diagnostiqué (Test
B) était réel et sa brique isolée se corrige bien (Test F, jusqu'à 1,68×) — mais
l'implémentation du correctif introduit un second défaut (le score max coûte plus cher
par appel que la moyenne qu'il remplace), et le bilan net, mesuré sur un vrai décodage
complet, est une **régression**, pas une amélioration. Le chantier n'est pas résolu ; il
est mieux caractérisé, avec un nouveau défaut précis à corriger (rendre le score max
aussi compact que la réduction logsumexp qu'il remplace) plutôt qu'une intuition vague de
"il faut fusionner le noyau".

---

## 9. Ce que ça change / ne change pas

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
- **Confirmé et mesuré (§7)** : le correctif (tampons incrémentaux + score max) accélère
  réellement la construction des résumés — jusqu'à **1,68× à 262k tokens**, un gain qui
  croît avec le contexte comme prédit. Reste à intégrer dans le pipeline complet et
  mesurer l'effet bout en bout sur un vrai passage de modèle multi-pas — pas encore fait,
  mais la brique qui manquait est maintenant validée isolément.

## État final — où en est ASP après cette session (3 sessions GPU, terminées)

1. **Le résultat négatif du papier tient et se confirme encore** : ASP, dans toutes les
   variantes testées (original et corrigée), est plus lent que dense en décodage complet.
   Confirmé sur 2 GPU indépendants (H200, RTX 4090), et maintenant sur un vrai décodage
   multi-pas, pas seulement des instantanés figés.
2. **Deux causes réelles ont été isolées et comprises** : reconstruction complète des
   résumés à chaque pas (§6, Test B) et score par moyenne au lieu de max (validé
   indépendamment en qualité). Chacune se corrige **isolément** avec un gain net
   (§7 : jusqu'à 1,68× sur la seule construction des résumés).
3. **Mais l'intégration complète des deux correctifs régresse** (§8) : le score max, une
   fois utilisé pour trier *tous* les blocs à chaque pas (pas seulement pour construire
   les résumés), coûte lui-même plus cher par appel que la réduction compacte qu'il
   remplace — un **troisième défaut**, de nature différente des deux premiers, découvert
   en testant le correctif plutôt qu'en l'implémentant à l'aveugle.
4. **Bilan honnête** : trois sessions GPU (< 10 $ au total) ont transformé un résultat
   négatif brut ("c'est plus lent, point") en un résultat négatif *caractérisé* — chaque
   défaut trouvé a un nom, une cause et une piste de correction précise, mais aucune
   combinaison testée à ce jour ne rend ASP plus rapide que dense sur un vrai modèle.
   Ce n'est pas un échec de la démarche : c'est le type de résultat qu'une vraie
   investigation produit quand la réponse honnête est "non, pas encore".

## Reproductibilité

- `16_gpu/bench_gather.py` / `colab_driver.py` (gather), `e2e_qwen.py` (bout en bout,
  version originale/buguée), `e2e_qwen_v2.py` (version corrigée, tampons incrémentaux +
  score max), `vent.py` (ventilation, bande passante configurable via `ASP_BW`),
  `probe_b_decomposition.py` / `probe_c_batch.py` / `probe_d_incremental.py` (raté) /
  `probe_f_multistep_v2.py` (refait proprement).
- `12_poc/code/capture_qk.py` (patché GPU bf16 + fallback dataset `Salesforce/wikitext`),
  `eval_selection.py --tag qwen8b_gpu`.
- `19_loi_octets/code/frontiere2.py --tags qwen8b_gpu` (patché chemins portables).
- `21_rtx4090_reel/test_a_stabilite_selection.py`,
  `21_rtx4090_reel/test_e_mean_vs_max_premier_etage.py` (gratuits, locaux, sans GPU).
- Dépôt complet (code, sans les gros binaires) : github.com/Alicienn/inference-ai.
