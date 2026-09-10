# Recadrage — à coller dans la session en cours

Le travail théorique (ARCHETYPES, la théorie à deux canaux, le Théorème 9) est solide et
dans l'esprit demandé — continue sur cette lancée. Trois points précis avant d'aller plus
loin en théorie :

## 1. Priorité absolue : exécute le protocole de falsification que tu as toi-même écrit

`RAPPORT_allocation_precision.md` §5 donne un protocole à coût nul, prêt à lancer
(SmolLM2-135M, Qwen2.5-0.5B, masque causal vérifié). Tu as maintenant l'exécution. Lance-le
avant d'ajouter la moindre nouvelle couche théorique. Le test est binaire par construction
(`AM/GM ≈ 1` → technique morte) — tu n'as pas besoin d'un résultat parfait, juste d'un
chiffre. `code/` et `resultats/` sont vides depuis le début de la session ; c'est le signal
qu'il est temps de mesurer plutôt que de démontrer.

## 2. Recoupe avec ce que ce projet a déjà mesuré, avant de mesurer à nouveau

Le Théorème 9 prédit un gain d'allocation nettement plus grand que ce que ce projet a déjà
trouvé empiriquement dans `19_loi_octets/RAPPORT_OCTETS.md` (« l'allocation de bits vaut
~2 bits gratuits », mesuré sur Qwen3-8B/SmolLM2, RoPE, par bloc de fréquence — pas par
token). Avant de lancer une nouvelle capture, vérifie si tes propres calculs, appliqués aux
données déjà capturées dans `12_poc/resultats/qk_qwen8b_gpu.npz` (Qwen3-8B réel, RoPE,
8192 tokens) ou dans `19_loi_octets/resultats/`, donnent un `Δb` cohérent avec ce ~2 bits
déjà établi, ou si ta théorie prédit vraiment plus (et pourquoi le projet précédent ne
l'aurait pas vu). Les deux réponses sont intéressantes ; ne pas les comparer serait un trou
méthodologique.

## 3. Anticipe le problème de lecture hétérogène AVANT qu'il ne tue la piste comme ASP

Tu as toi-même noté le risque (« un cache KV à largeurs hétérogènes perd la
vectorisation ») mais tu ne l'as pas encore transformé en design constraint. Le projet a
déjà la réponse partielle : `16_gpu/RAPPORT_GATHER.md` montre que la lecture dispersée est
gratuite dès que chaque morceau lu dépasse ~1-2 Kio (mesuré sur 4 GPU). Réponds à cette
question avant d'aller plus loin : **si l'allocation de précision variable est appliquée
par bloc (pas par token individuel), avec des blocs assez gros pour respecter ce seuil
de ~1 Kio, le gain théorique survit-il, et de combien se réduit-il** (puisque grouper des
tokens de sensibilité différente dans un même bloc à largeur uniforme dilue l'allocation
idéale) ? C'est un calcul, pas une mesure — fais-le avant de concevoir un format de
stockage.

Une fois ces trois points traités, tu peux reprendre les pistes secondaires ouvertes
(MoE `½log₂k_eff`, gradients/all-reduce avec error feedback) — elles restent valides et
pas chères à tester.
