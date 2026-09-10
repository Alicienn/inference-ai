# Prompt de lancement — nouvelles techniques de réduction de coût (inférence + entraînement)

*Document généré le 10 septembre 2026, à coller tel quel dans l'IA de recherche.*

---

## Contexte

Tu travailles sur `C:\Users\alici\Downloads\inference_opti`, le même projet que celui sur
lequel tu as mené la recherche sur ASP (sélection de blocs à deux passes) documentée dans
`20_sortie_attention/`, `21_rtx4090_reel/` et `17_papier/asp.tex`. Cette piste est
**close** : trois sessions GPU réelles (H200 puis RTX 4090 ×2) ont confirmé qu'ASP
économise des octets mais ne devient jamais plus rapide que l'attention dense, même après
correction de deux défauts distincts. Tu n'as plus à y revenir sauf si une idée nouvelle
en découle naturellement.

Le corpus déjà constitué (64 papiers arXiv classés dans `01_deepseek/` à `18_refs_papier/`,
indexé dans `INDEX.md`) reste disponible et pertinent. Le fichier
`C:\Users\alici\Downloads\DeepSeek-V4.1-Flash_resume_technique.md` (joint) résume un
rapport technique sorti aujourd'hui sur DeepSeek-V4.1-Flash. Ce n'est **pas** la cible à
reproduire — c'est un **exemple de démarche** à comprendre puis dépasser.

## Ce qu'il faut extraire du document DeepSeek : la méthode, pas seulement le résultat

Lis ce document non pas comme une liste de techniques à copier, mais comme un catalogue
d'**archétypes de preuve** qui ont chacun permis de transformer une intuition en gain
mesuré et défendable. Identifie et nomme ces archétypes avant de chercher tes propres
idées. Quelques exemples déjà présents dans le document, pour te donner le niveau
d'exigence attendu — pas une liste fermée :

- **Argument de borne numérique** (FP4 KV Cache) : montrer qu'une plage de représentation
  couvre largement la distribution réelle des valeurs (`2688 ≫ 22.6`), donc qu'une
  simplification (supprimer une échelle de second niveau) ne coûte rien en pratique.
- **Comptage de complexité asymptotique** (CED, Hierarchical Sparse Indexer) : réécrire le
  calcul pour qu'un terme domine moins (`O(NL) → O(NL/2)`, linéaire → constant par
  requête), avec preuve algébrique directe par substitution.
- **Comptage de trafic mémoire contre une borne inférieure théorique** (Single-Pass mHC) :
  établir la borne idéale de lectures/écritures, montrer l'écart de l'implémentation
  actuelle, puis un réarrangement de l'ordre des dépendances de données qui atteint
  exactement la borne.
- **Comptage empirique de kernels/lancements** (CSA2) : quand la preuve analytique ne
  suffit pas, mesurer directement le nombre d'opérations GPU réellement exécutées.
- **Dérivation locale sous hypothèse simplificatrice, validée ensuite empiriquement**
  (contrôle de l'effort RL) : poser un modèle simple, en tirer une prédiction testable
  (ici une relation affine), puis vérifier que la prédiction tient en pratique.
- **Ce que l'absence d'innovation révèle** (post-training) : parfois le vrai levier n'est
  pas algorithmique mais dans l'échelle et la rigueur de l'ingénierie autour (données,
  environnements, déduplication) — un résultat négatif en soi est une information.

Ta première tâche est donc analytique : dresser la liste complète de ces archétypes de
preuve tels qu'ils apparaissent dans le document, avec pour chacun la question générale
qu'il pose (« où est le terme qui domine et peut-il être rendu constant/partagé/plus
grossier sans perte ? », « quelle est la borne inférieure théorique de ce trafic et
qu'est-ce qui nous en éloigne ? », etc.). C'est cette boîte à outils méthodologique, pas
les techniques elles-mêmes, qui doit ensuite te servir à générer des idées neuves.

## Objectif — ambitieux, sans plafond fixé à l'avance

**Trouver, développer et valider au moins une technique nouvelle et non triviale qui
réduit le coût de calcul et/ou de temps de l'inférence et/ou de l'entraînement des grands
modèles de langage**, avec le même niveau d'exigence de preuve que le document
DeepSeek — analytique quand c'est possible, empirique et mesuré quand ce ne l'est pas,
jamais une simple intuition non vérifiée.

Aucune limite de périmètre n'est fixée a priori : attention, KV cache, MoE, quantification
des poids, optimiseur, ordonnancement de couches, parallélisme, formats numériques,
architecture, données d'entraînement, post-training, serving — tout est sur la table. Ne
te limite pas à améliorer ou combiner les techniques du document DeepSeek ; elles sont un
point de départ pour calibrer ton niveau d'ambition et de rigueur, pas un cahier des
charges. Une idée entièrement différente, y compris hors du champ de ce document, est
bienvenue si elle est bien fondée.

**Niveau d'ambition** : vise des gains structurels (division par un facteur constant du
travail asymptotique, pas quelques pourcents d'optimisation locale) — dans l'esprit de
« O(N) devient O(1) par requête » ou « le trafic mémoire est divisé par 2 exactement »,
pas dans l'esprit d'un réglage d'hyperparamètre. Une piste qui échoue mais dont l'échec
est bien caractérisé (comme la piste ASP) est un résultat valide et publiable — ne
filtre pas tes idées uniquement sur celles qui « marchent », mais consacre l'essentiel de
ton effort à celles qui survivent à un premier examen critique.

## Liberté d'action — aucune restriction d'outils ou de méthode

Utilise tous les outils à ta disposition sans qu'il soit besoin de les lister ici :
recherche web, exécution de code, calcul symbolique, lecture/écriture de fichiers,
recherche dans le corpus existant (`_tools/corpus_search.py` interroge `_txt/`),
n'importe quel modèle ou bibliothèque accessible en local. Si un outil ou un accès
supplémentaire te serait utile (accès à un GPU loué, un modèle particulier, un papier non
présent dans `_txt/`), demande-le explicitement plutôt que de t'en passer en silence — la
session de recherche associée à ce projet a déjà loué des GPU (RTX 4090 sur vast.ai) et
peut recommencer si une hypothèse le justifie clairement.

**Budget et rigueur expérimentale** : privilégie autant que possible la validation à coût
nul (CPU, petits modèles type GPT-2/SmolLM2-135M/Qwen2.5-0.5B, calcul analytique) avant de
demander du temps GPU — c'est la discipline déjà en place dans ce projet (voir comment le
test de stabilité de sélection, gratuit, a été fait avant tout test payant). Quand une
mesure GPU réelle devient nécessaire pour trancher, dis-le clairement et pourquoi, avec
une estimation du temps de calcul nécessaire.

## Standards de rigueur — hérités du reste du projet, non négociables

- Étiquette chaque affirmation **[FAIT]** (mesuré ou prouvé), **[INFÉRENCE]** (déduction
  raisonnée à partir de faits) ou **[HYPOTHÈSE]** (spéculatif, non vérifié) — convention
  déjà utilisée dans `07_analyses/SYNTHESE.md` et `INDEX.md`.
- Quand une hypothèse initiale est infirmée par la mesure, dis-le explicitement et
  explique pourquoi tu te trompais — ne réécris pas l'historique. Le journal
  `20_sortie_attention/RAPPORT_v28.md` (et ses versions archivées dans
  `20_sortie_attention/_archive_versions/`) est l'exemple à suivre : plusieurs
  autocorrections en cascade, chacune documentée plutôt que masquée.
- Vérifie indépendamment tes résultats importants avant de les considérer acquis (une
  réimplémentation séparée qui reproduit le chiffre, comme cela a été fait pour le
  sélecteur ASP) plutôt qu'une seule mesure.
- Distingue toujours la **qualité** (précision, perte, rappel) de la **vitesse** (latence,
  débit) — ne conclus jamais qu'une technique est bonne sur un seul de ces deux axes sans
  vérifier l'autre. C'est l'erreur qui a fait perdre du temps sur ASP (un correctif validé
  en qualité s'est révélé négatif en vitesse une fois mesuré).

## Architecture documentaire — un cadre, pas un carcan

Crée un nouveau dossier numéroté à la racine du projet, **`22_nouvelles_techniques/`**
(suite logique de la numérotation existante : 01-15/18 = corpus, 16-21 = travail sur ASP).
À l'intérieur, structure libre tant que ces quelques repères sont respectés :

```
22_nouvelles_techniques/
  PISTES.md              journal de bord libre : idées explorées, écartées, pourquoi —
                          pas besoin d'être poli ou structuré, c'est ton espace de travail
  code/                  tout script de validation, quel que soit son état de finition
  resultats/              sorties brutes (JSON, txt, logs) référencées par les rapports
  RAPPORT_<nom>.md        un rapport par technique qui atteint un résultat assez solide
                          pour être cité (positif OU négatif bien caractérisé) — libre
                          dans sa forme, mais doit permettre à quelqu'un d'autre de
                          reproduire ta mesure (quel script, quelle commande, quelles
                          données) et de retrouver tes chiffres exacts
  SYNTHESE.md             à produire en fin de parcours (ou à mettre à jour au fil de
                          l'eau) : vue d'ensemble de ce qui a été exploré, ce qui tient,
                          ce qui ne tient pas, et lequel des résultats mérite le plus
                          d'attention pour la suite
```

Ne te sens pas obligé de produire un `RAPPORT_*.md` pour chaque piste explorée — beaucoup
d'idées s'écarteront après quelques minutes de réflexion ou de calcul et n'ont leur place
que dans `PISTES.md`. Un rapport formel se justifie quand tu as quelque chose d'assez
solide (prouvé ou mesuré, avec une marge de confiance claire) pour que ça compte comme un
résultat du projet, pas un brouillon.

Une fois une ou plusieurs pistes bien avancées, ajoute une entrée dans `INDEX.md` à la
racine (section « Axe 22 », suivant le même format que les entrées Axe 20/21 déjà
présentes) pointant vers `22_nouvelles_techniques/SYNTHESE.md`.

## Point de départ, si utile (non obligatoire)

Si tu veux un premier point d'ancrage plutôt que de partir d'une page blanche, ces
questions ouvertes sont cohérentes avec l'esprit du document DeepSeek et n'ont pas encore
de réponse claire dans ce projet ou dans le corpus déjà lu :

- Le document DeepSeek partage le KV et les indices Top-K **entre couches** (CSA2). Existe-t-il
  un principe symétrique côté **entraînement** — partager un calcul coûteux entre couches
  adjacentes d'un même forward/backward, au-delà du weight tying classique ?
- Le partage FP4 sans échelle globale (borne numérique très généreuse) suggère que
  d'autres composants du pipeline (gradients ? états d'optimiseur ? activations
  intermédiaires du MoE ?) ont peut-être une marge de précision inexploitée, démontrable
  par le même argument de borne.
- La preuve de trafic mémoire du mHC single-pass repose sur un décalage d'un cran d'une
  dépendance de données pour permettre le pipelining. Où ailleurs dans un transformer
  standard une dépendance séquentielle inutile bloque-t-elle un kernel fusionné ?

Mais ne t'y limite pas — si une lecture plus large du corpus ou une intuition propre
mène ailleurs, suis cette piste.

## Ce que tu dois me rapporter, et quand

Ne m'attends pas pour explorer — avance en autonomie. Reviens vers moi (ou vers la session
qui a mené la recherche ASP, si tu peux la contacter) quand : (a) une piste atteint un
résultat assez solide pour mériter un `RAPPORT_*.md`, (b) tu as besoin d'un accès GPU ou
d'une ressource externe, ou (c) tu es bloqué sur une question dont la réponse engage la
direction de toute une piste.
