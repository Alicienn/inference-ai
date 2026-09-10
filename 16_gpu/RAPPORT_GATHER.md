# Le gather irrégulier annule-t-il le gain d'ASP ? — réponse mesurée

**Verdict : non — et de moins en moins a mesure que le GPU grossit.**
Mesure sur **trois** GPU reels : `gfx1152` (iGPU AMD, LPDDR5) penalite ~5 % ; **Tesla T4**
(GDDR6) ~1 % ; **A100 PCIe** (HBM2e) **0 %**, aucun seuil de granularite detectable.
La question initiale est tranchee. Un **second obstacle** la remplace sur gros GPU : le
cout de lancement de la seconde passe, qui exige un volume de travail suffisant
(~64 Mo sur A100) — et qui est un artefact d'implementation, non d'ASP (section 6ter).

---

## 1. Comment le blocage a été levé

Le blocage était formulé comme « il faut un noyau CUDA, donc un GPU NVIDIA ». En
sondant la machine j'ai constaté qu'aucune pile ROCm/HIP n'était installée, mais que
**le pilote Adrenalin expose déjà l'iGPU en OpenCL 2.1, sous le nom exact `gfx1152`**.
OpenCL suffit entièrement à mesurer des motifs d'accès mémoire : la question a donc pu
être tranchée **sans aucune installation**, en réutilisant ce qui était déjà là.

| Élément sondé | Constat |
|---|---|
| `hipcc`, `rocminfo`, `hipconfig`, `rocm-smi` | absents |
| `clinfo` + pyopencl | **présents**, device GPU `gfx1152` exposé |
| Caractéristiques | 4 WGP, 3 GHz, 12,2 Gio partagés, ligne de cache 64 o |

---

## 2. L'argument analytique, posé avant la mesure

[DÉRIVATION] La question est souvent mal posée. Le gather d'ASP n'a pas une
granularité de 4 octets : chaque élément gathéré est **un résumé de bloc entier**, soit
`r × D × 2` octets en fp16 — 1024 octets pour `r=8, D=64`, c'est-à-dire **16 lignes de
cache de 64 octets**. La coalescence fonctionne donc parfaitement *à l'intérieur* de
chaque morceau ; seule la localité *entre* morceaux est perdue.

Prédiction : la pénalité doit s'annuler dès que la taille de morceau dépasse largement
la ligne de cache. C'est ce que la mesure confirme, et elle en donne le seuil exact.

---

## 3. Mesure 1 — pénalité du gather selon la granularité

Deux noyaux au **flux d'instructions identique**, ne différant que par le motif
d'offsets ; volume lu identique (96 Mio) ; buffer source 768 Mio pour rester borné par
la DRAM.

| taille de morceau | débit contigu | débit gather | **ratio** |
|---:|---:|---:|---:|
| 64 o | 5,0 GB/s | 3,3 GB/s | **0,67** |
| 128 o | 9,7 | 6,7 | 0,69 |
| 256 o | 17,7 | 12,9 | 0,73 |
| 512 o | 29,1 | 24,1 | 0,83 |
| **1 024 o** | 38,7 | 30,2 | **0,78–1,07** |
| **2 048 o** | 59,3 | 59,9 | **1,01** |
| 4 096 o | 64,1 | 69,8 | 1,09 |
| 16 384 o | 78,7 | 77,6 | 0,99 |

[FAIT] **La bascule se situe entre 1 et 2 Kio.** En dessous, le gather coûte 20 à 35 % ;
au-dessus, il est gratuit.

---

## 4. Mesure 2 — le test décisif, motif ASP complet

Les deux schémas sont implémentés **au mieux de ce que leur motif permet** :
les lectures contiguës (plat, et passe 1 d'ASP) par un noyau de flux à segments de
16 Kio ; seule la passe 2 d'ASP garde un work-group par bloc, son irrégularité étant
irréductible.

[FAIT] 64 configurations (n ∈ {4096, 16384}, D ∈ {64, 128}, r1 ∈ {2,4}, r2 ∈ {8,16,32},
m = n/4 et n/8), deux balayages complets :

| grandeur | valeur |
|---|---|
| **efficacité médiane** (gain réel / gain théorique en octets) | **96,2 %** |
| étendue | 78,2 % – 115,0 % |
| configurations plus lentes que le plat | **0 / 64** |
| débit du gather | 75,7 GB/s (médiane) |
| débit du contigu | 79,7 GB/s (médiane) |
| **gather / contigu** | **94,9 %** |
| contrôle de dérive thermique (2ᵉ balayage) | 94,0 %, écart 2,3 % |

Exemples de gains réels : `n=16384, D=64, r1=4, r2=32, m=2048` → gain théorique 4,00×,
**gain réel 4,60×** ; `r1=2, r2=16, m=4096` → théorique 2,67×, réel 2,42×.

Les valeurs supérieures à 100 % sont du bruit de mesure sur iGPU partagé, pas un gain
supra-linéaire : il ne faut pas les lire comme telles.

### Le détour instructif

[AUTOCORRECTION] Une première version du banc donnait une efficacité médiane de
**46,7 %** et concluait à un gather coûteux. Le diagnostic par phase a montré que la
passe 2 (le gather) tournait déjà à 52–85 GB/s — la vitesse du contigu — et que le
goulot était la **passe 1**, à 13–53 GB/s. Or la passe 1 est un accès **contigu** :
elle était simplement mal implémentée (un work-group par bloc de 128 à 512 octets,
régime où le GPU ne sature pas). C'était un artefact du banc, pas une propriété d'ASP.
Sans ce diagnostic par phase, j'aurais conclu l'inverse de la vérité.

---

## 5. Frontière de validité — une règle de conception

[FAIT] Efficacité d'ASP selon la taille du résumé fin `S2 = r2 × D × 2` :

| D | r2 | S2 | gain théorique | gain réel | efficacité |
|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 512 o | 1,33× | **0,99×** | 74 % |
| 64 | 8 | 1 024 o | 2,00× | 1,74× | 87 % |
| 64 | 16 | 2 048 o | 2,67× | 2,02× | 76 % |
| 128 | 4 | 1 024 o | 1,33× | 1,37× | 103 % |
| 128 | 8 | 2 048 o | 2,00× | 1,86× | 93 % |
| 128 | 32 | 8 192 o | 3,20× | 3,17× | 99 % |

> **Règle provisoire (iGPU seul) : `r2 × D × 2 ≥ 1024 octets`.**
> En dessous, le morceau gathéré ne couvre pas assez de lignes de cache et le gain
> disparaît (0,99× à 512 octets).
>
> ⚠️ **Révisée à 2048 o après la mesure T4** — voir §6bis, qui ajoute aussi une seconde
> contrainte de volume total. Cette section reste telle quelle comme trace de ce que
> l'iGPU seul permettait de conclure.

---

## 6. Ce que ce test valide, et ce qu'il ne valide pas

**Il valide — le mécanisme.** Un gather dont la granularité est un bloc de ≥ 1 Kio
préserve la coalescence et atteint 95 % du débit contigu. Le gain théorique en octets
d'ASP se convertit donc presque intégralement en gain de temps mural. Ce mécanisme
(taille de morceau ≫ ligne de cache) est architecture-indépendant dans son principe.

**Il ne valide pas — les chiffres absolus en production.** Le 860M est un iGPU RDNA 3.5
à 4 WGP sur LPDDR5 **partagée avec le CPU**, à ~80 GB/s. Un H100 ou un MI300 a de la
HBM à ~3,3 To/s (facteur **≈ 40**), 100+ unités de calcul, des pages DRAM et une
hiérarchie de caches différentes, et exige bien plus de parallélisme pour saturer. Trois
réserves précises :

1. ~~**Sur HBM, le seuil de 1 Kio pourrait se déplacer**~~ — **partiellement résolu** :
   la mesure T4 (§6bis) donne le même seuil de 1 Kio sur GDDR6, ce qui suggère une origine
   DRAM (invariante) plutôt que MLP (variable). HBM reste non testé.
2. **La mémoire partagée avec le CPU** introduit une variabilité absente d'un GPU
   discret ; c'est pourquoi le double balayage de contrôle a été fait (écart 2,3 %).
3. **Le banc isole le motif d'accès mémoire**, pas un noyau d'attention complet
   (calcul des scores, softmax, occupation des registres). Dans un noyau réel, la
   latence du gather peut être mieux ou moins bien recouverte par le calcul.

**Réserve levée pour NVIDIA/GDDR6** par la mesure T4 du §6bis. Pour HBM, il reste à
lancer `colab run --gpu A100 colab_driver.py` (voir [`GUIDE_SETUP.md`](GUIDE_SETUP.md)).

---

## 6bis. Confirmation sur Tesla T4 (NVIDIA, GDDR6) — et une prediction refutee

[FAIT] Banc execute sur un **Tesla T4** loue via `google-colab-cli` (40 SM, bus 256 bits,
pic mesure 274 GB/s), source CUDA compilee par `nvcc -arch=sm_75`.

### Le resultat central tient

| morceau | contigu | gather | ratio | sature ? |
|---:|---:|---:|---:|---|
| 64 o | 14,5 GB/s | 14,5 | 1,002 | non - borne par le lancement |
| 256 o | 83,1 | 75,5 | 0,908 | partiel |
| 512 o | 167,7 | 147,6 | 0,880 | partiel |
| **1024 o** | 235,3 | 232,5 | **0,988** | **oui** |
| 2048 o | 274,0 | 260,0 | 0,949 | oui |
| 4096 o+ | ~270 | ~269 | 0,99-1,00 | oui |

**En regime sature, le gather atteint 99,1 % du debit contigu** (mediane), contre 94,9 %
sur l'iGPU AMD. Le mecanisme est confirme sur une seconde architecture memoire.

### Ce qui est refute

[REFUTE] Le modele MLP (loi de Little) predisait, **avant la mesure**, un seuil de
**328 o** sur T4 - plus bas que les 1024 o de l'iGPU. **Mesure : 1024 o sur T4 aussi.**
La calibration du modele est fausse d'un facteur ~3 ; seule sa direction survit (a 512 o,
T4 fait 0,880 contre 0,83 sur l'iGPU, donc legerement mieux).

[INFERENCE] Le seuil vaut ~1 Kio sur deux systemes tres differents (LPDDR5 partagee
80 GB/s / 8 CU, et GDDR6 274 GB/s / 40 SM). Cette **invariance** est ce que predit un
argument de granularite **DRAM** (pages et bursts de l'ordre du Kio dans les deux cas),
pas la loi de Little qui annoncait un facteur 3. J'avais mentionne cet effet DRAM puis
l'avais ecarte en le supposant absorbe par le parallelisme de bancs - c'etait l'erreur.

**Consequence favorable :** la regle `r2 x D x 2 >= 1024 o` n'a pas a etre relevee a 2 Kio
sur HBM, contrairement a ce que le modele MLP suggerait. HBM reste toutefois non teste.

### Un second facteur, invisible sur l'iGPU

[FAIT] Efficacite ASP sur T4 : **mediane 79,2 %**, 2/32 configurations plus lentes.

| taille du resume S2 | efficacite | volume total lu | efficacite |
|---:|---:|---|---:|
| 1024 o | **50,0 %** | < 8 Mo | **28,3 %** |
| 2048 o | 77,2 % | 8-64 Mo | 78,4 % |
| 4096 o | 85,2 % | > 64 Mo | **85,2 %** |
| 8192 o | **90,3 %** | | |

[INFERENCE] **Deux contraintes, pas une.** Sur un gros GPU, la structure a deux noyaux
d'ASP paie un double cout de lancement qu'il faut amortir - ce que l'iGPU ne revelait pas,
saturant avec bien moins de travail. Regles revisees :

> **(1)** `r2 * D * 2 >= 2048 octets` (1024 o ne donne que 50 % d'efficacite sur T4)
> **(2)** `n * S2 >= 8 Mo` de volume lu par passe

Le regime de deploiement reel les satisfait largement : a 1M de contexte avec L=64,
n = 15 625 blocs, et avec r2=16, D=128 on a S2 = 4096 o pour 64 Mo de volume. Les echecs
se concentrent sur les petits n - les contextes courts, ou ASP n'a de toute facon aucun
interet. **Seuil d'utilite estime : ~130 k tokens** sur un GPU de classe T4.

---

## 6ter. A100 PCIe 40 Go (HBM2e) — le gather est GRATUIT, et mes deux modeles tombent

[FAIT] Banc execute sur une instance vast.ai **A100-PCIE-40GB** (CUDA 12.8, sm_80, 108 SM,
bus 5120 bits, pic mesure **1372 GB/s**).

### Aucune penalite de gather, a aucune granularite

| morceau | contigu | gather | ratio | % du pic | regime |
|---:|---:|---:|---:|---:|---|
| 64 o | 55,3 | 55,3 | 1,000 | 4 % | borne par le lancement |
| 512 o | 438,4 | 438,4 | 1,000 | 32 % | partiel |
| 1024 o | 862,3 | 848,4 | 0,984 | 63 % | partiel |
| 2048 o | 1272,5 | 1236,5 | 0,972 | 93 % | sature |
| 4096 o+ | ~1370 | ~1340 | 0,955-1,031 | ~100 % | sature |

**Ratio minimum sur tout le balayage : 0,955.** En dessous de 512 o la mesure n'est pas
informative (ni l'un ni l'autre ne sature). A charge relative comparable (32 % du pic),
le T4 donnait 0,880 et l'A100 donne 1,000.

### Deux refutations

[REFUTE] **Modele MLP** : predisait `C* = 1009 o` sur A100. Aucun seuil observe au-dessus
de 512 o. Faux quantitativement pour la seconde fois (328 o predits sur T4, 1024 observes).

[REFUTE] **Invariance DRAM**, hypothese adoptee apres le T4 : le seuil serait ~1 Kio
partout. Il **disparait** sur A100. Il n'est donc pas invariant.

[INFERENCE] Ce qui survit : la **direction** du modele MLP. La penalite decroit
monotonement avec le parallelisme —
gfx1152 (8 CU) 0,67 a 64 o · T4 (40 SM) 0,88 a 512 o · A100 (108 SM) 1,00 a 512 o.
**Sur GPU datacenter, le gather d'ASP est gratuit.** La question initiale est tranchee.

### Le vrai obstacle a change de nature

[FAIT] Efficacite ASP brute sur A100 : **mediane 30,6 %, 16/32 plus lentes** que le plat.

[DERIVATION] `t_p1` et `t_p2` ne descendent jamais sous **11,3 us** : c'est le cout de
lancement d'un noyau. Or le schema plat ne prend que 12 a 30 us dans la plupart des
configurations — le second lancement d'ASP suffit a tout annuler.

| volume total lu | efficacite brute |
|---|---:|
| < 16 Mo | 22,2 % |
| 16-64 Mo | 30,6 % |
| **> 64 Mo** | **77,6 %** |

Meilleure configuration : 134 Mo -> **94,2 %** d'efficacite, gain reel 2,51x.

[LIMITE] **Mon banc mesure ASP comme deux lancements de noyau separes.** Une
implementation reelle fusionnerait les deux passes (noyau unique, graphe CUDA ou noyau
persistant). Le cout de lancement est donc un artefact du banc, pas une propriete d'ASP —
mais je n'ai pas mesure de version fusionnee, donc **je ne peux pas affirmer que le gain
est recupere**. C'est le prochain test.

[ABANDON] Correction analytique tentee (retrancher le plancher de 11,3 us aux deux
schemas) : **inutilisable**, car pour les petites configurations le temps total EST
l'overhead — la soustraction produit des efficacites absurdes (9413 %). Chiffres bruts
conserves, artefact signale.

### Bilan sur quatre GPU

| | gfx1152 (LPDDR5) | T4 (GDDR6) | A100 (HBM2e) | RTX 4090 (GDDR6X) |
|---|---:|---:|---:|---:|
| pic mesure | 80 GB/s | 274 GB/s | 1372 GB/s | 933 GB/s |
| penalite de gather | ~5 % | ~1 % | **0 %** | **~1 % (min 0,969)** |
| seuil de granularite | ~1-2 Kio | ~1 Kio | **aucun >=512 o** | **aucun >=512 o** |
| efficacite ASP mediane (toutes lignes) | 96,2 % | 79,2 % | 30,6 % (brute) | 47,0 % |
| efficacite ASP, lignes <= L2 | — | 28,3 % | 30,2 % | 33,5 % |
| efficacite ASP, lignes > L2 | — | 81,0 % | 75,9 % | 250,3 % (2 lignes) |
| volume requis par passe | quelques Mo | ~8 Mo | **~64-134 Mo** | ~34-50 Mo |

---

## 6quater. RTX 4090 (GDDR6X) — quatrieme architecture, et un artefact de mesure demasque

[`resultats/rtx4090_raw.txt`](resultats/rtx4090_raw.txt) — NVIDIA GeForce RTX 4090,
49140 Mio, sm_89, 128 CU/SM, bus 384 bits, 2520 MHz.

### Le seuil de granularite ne reapparait pas

Pic contigu mesure **933,2 GB/s**. La saturation (>=80 % du pic) est atteinte des
**512 o** (95 % du pic), et au-dela de 512 o le ratio gather/contigu vaut
0,969 / 0,972 / 0,992 / 0,993 / 0,996 / 1,000 / 0,997 pour 512 o a 32 Kio :
**moyenne 0,9884, minimum 0,969**. Sous 512 o le chemin contigu lui-meme
n'atteint que 13 % / 27 % / 54 % du pic : on y mesure du cout de lancement, pas de
la memoire. Quatrieme architecture, quatrieme fois : aucun seuil a ~1 Kio ; sur la
4090 il est a **512 o au plus**, comme sur l'A100 il avait disparu.

### Le chiffre d'efficacite ne mesure pas la DRAM mais le cache

Le banc publie aussi une « efficacite » : gain reel / gain en octets. Sur la 4090
elle vaut **47,0 % en moyenne** et **20 des 32 configurations sont plus lentes**
que le chemin plat. Les deux configurations au-dessus de 200 % (278,4 % et
222,3 %) sont les **seules honnetes** : le chemin plat lit `n x r2 x D x 2`
octets, volume qui ne depasse les 72 Mo de L2 de la carte que pour ces deux
lignes (128 Mo).

Preuve independante du diagnostic : le debit plat **implique** par les 30 autres
lignes va de **822 a 3647 GB/s** — impossible en DRAM quand le pic mesure sur la
meme carte est de 933 GB/s. Ces lignes sont servies par la L2 ; le « gain » du
chemin plat y est un gain de cache, que le chemin ASP (dont le volume ne depasse
jamais 50 Mo) ne reproduit pas dans les memes proportions.

Le meme partage de regime apparait sur les trois autres GPU :

| GPU | L2 | lignes <= L2 | efficacite <= L2 | lignes > L2 | efficacite > L2 |
|---|---:|---:|---:|---:|---:|
| Tesla T4 | 4 Mo | 2 | 28,3 % | 30 | 81,0 % |
| A100 PCIe | 40 Mo | 24 | 30,2 % | 8 | 75,9 % |
| RTX 4090 | 72 Mo | 30 | 33,5 % | 2 | 250,3 % |

Sur le T4 et l'A100 le volume et le regime de cache varient ensemble : le partage
ne peut pas y etre attribue au seul cache. Sur la 4090, si, parce que le debit
implicite depasse le pic DRAM mesure.

### Ce qui reste valable, et ce qui doit etre retire

Valable : la penalite de gather (le balayage lit 96 Mio par lancement, au-dessus
de toutes les L2 de l'etude) ; la regle `r2 x D x 2 >= 1024 o` (sur la 4090 elle
est conservatrice) ; l'absence de seuil a 1 Kio.

A retirer des tableaux de synthese : le classement d'efficacite 96,2 % / 79,2 % /
30,6 %, qui melange des regimes de cache differents. A remplacer par : efficacite
~30 % quand le volume plat tient en L2, 76-81 % quand il en sort, et **aucune des
32 configurations ne compare deux chemins tous deux en DRAM**.

---

## 7. Contre-point CPU

[SIGNAL FAIBLE — non transposable au GPU] [`proxy_cpu.py`](proxy_cpu.py) montre la même
tendance qualitative sur une hiérarchie mémoire entièrement différente : ratio
gather/contigu de 0,58 (morceaux de 256 o) à 0,98 (64 Kio). Les mécanismes diffèrent
(prefetcher matériel et lignes de 64 o côté CPU ; coalescence de wavefront et pages DRAM
côté GPU) : ce résultat ne vaut pas validation, seulement contre-point.

---

## 8. Conséquence pour ASP

Le gain en octets mesuré dans le POC (médiane 1,24×–1,64×, meilleures configurations
1,7×–2,4×) survit au gather **à 96 % près**, sous réserve de respecter la règle
`r2 × D × 2 ≥ 1024 o`. La confiance dans ASP passe donc de « moyenne, non validée en
temps réel » à **« élevée sur le mécanisme, moyenne sur les chiffres en production »**.

Fichiers : [`bench_gather.py`](bench_gather.py) · [`bench_asp_v2.py`](bench_asp_v2.py) ·
[`asp_bench.cpp`](asp_bench.cpp) · [`proxy_cpu.py`](proxy_cpu.py) ·
résultats JSON dans [`resultats/`](resultats/).
