# Méthodes, hypothèses et limites

Ce document permet de reproduire et de contester tous les chiffres produits dans ce dossier.

## 1. Environnement

- Windows 11, **aucun GPU** — tout est analytique ou exécuté sur CPU.
- Python 3.11.1, numpy 2.4.4, torch 2.12.0+cpu, transformers 5.9.0, pypdf.
- Modèles utilisés : **GPT-2** (117 M, cache local) pour les cartes d'attention ;
  **SmolLM2-135M** (30 couches, GQA 9/3, head_dim 64, RoPE base 10⁴) pour les activations Q/K.
- Corpus texte : `wikitext-103-raw-v1`, split `test`.

## 2. Scripts et ce qu'ils produisent

| Script | Produit | Sortie brute |
|---|---|---|
| [`_tools/model_inference_cost.py`](../_tools/model_inference_cost.py) | Paramètres, FLOPs/token, KV cache pour V3.2 / V4-Flash / V4-Pro | `_sorties_brutes/01_*` |
| [`_tools/roofline_decode.py`](../_tools/roofline_decode.py) | Octets lus/token, intensité arithmétique, capacité | `_sorties_brutes/02_*` |
| [`_tools/exp_A2_mechanism.py`](../_tools/exp_A2_mechanism.py) | Autocorrélation, aiguilles, masse par granularité | `_sorties_brutes/03_*` |
| [`_tools/exp_A3_jensen_gap.py`](../_tools/exp_A3_jensen_gap.py) | Trou de Jensen selon la piquosité β | `_sorties_brutes/03b_*` |
| [`_tools/exp_B_rope_bit_allocation.py`](../_tools/exp_B_rope_bit_allocation.py) | Énergie par bloc RoPE, allocation vs uniforme | `_sorties_brutes/04_*` |
| [`_tools/exp_D_serving_economics.py`](../_tools/exp_D_serving_economics.py) | Transfert KV en désagrégation | `_sorties_brutes/05_*` |
| [`_tools/exp_C_spec_x_sparse.py`](../_tools/exp_C_spec_x_sparse.py) | Recouvrement des sélections, spéculation × sparsité | `_sorties_brutes/06_*` |
| [`_tools/exp_E_crossvalidation.py`](../_tools/exp_E_crossvalidation.py) | **Validation croisée** des résultats A2/A3/C sur SmolLM2-135M | `_sorties_brutes/07_*` |
| [`_tools/corpus_search.py`](../_tools/corpus_search.py) | Recherche dans le texte extrait des PDFs | — |

Reproduction :

```bash
cd C:\Users\alici\Downloads\inference_opti\_tools && python model_inference_cost.py
```

## 3. Sources des hyperparamètres

| Modèle | Source |
|---|---|
| DeepSeek-V3 / V3.2 | 2412.19437 §4.2 (61 couches, d=7168, MLA n_h=128, d_h=128, d_c=512, d'_c=1536, d_R=64 ; MoE 1+256, 8 actifs, d_ff=2048) ; 2512.02556 (DSA top-k=2048) |
| DeepSeek-V4-Flash | 2606.19348 §4.2.1 (43 couches, d=4096, CSA m=4 top-k=512, HCA m'=128, n_h=64, c=512, d_c=1024, g=8, d_g=1024, n_win=128 ; MoE 1+256, 6 actifs, d_ff=2048 ; mHC n_hc=4) |
| DeepSeek-V4-Pro | 2606.19348 §4.2.1 (61 couches, d=7168, top-k=1024, n_h=128, d_c=1536, g=16 ; MoE 1+384, 6 actifs, d_ff=3072) |

## 4. Hypothèses explicites (et leur effet)

Ces choix ne sont pas donnés par les papiers. Chacun est signalé avec son sens d'influence.

1. **Nombre de têtes de l'indexeur DSA de V3.2 = 64, dim 128.** Le papier V3.2 ne le précise pas ;
   j'ai retenu la configuration publique de V3.2-Exp, identique à celle de V4. *Effet :* c'est la
   principale source de l'écart résiduel sur les FLOPs de V4-Flash (8,43× reconstruit contre 9,8×
   annoncé). Doubler ce nombre porterait la baseline V3.2 à ~2,1 TFLOP et le ratio à 16×.
2. **Répartition des couches V4-Flash : 20 CSA / 21 HCA** (41 couches alternées après 2 couches SWA).
   *Effet :* ±1 couche déplace le KV cache d'environ 4 %.
3. **Précision du cache des clés d'indexation : FP4 pour V4** (cohérent avec « l'attention dans le
   lightning indexer est calculée en FP4 », §2.3.4), **FP8 pour V3.2**. *Effet :* c'est
   l'hypothèse la plus sensible du calcul KV. En FP8 pour V4, le ratio tomberait de 14,2× à ~12,3×.
   Le fait que FP4 reproduise le 13,7× annoncé est un argument *a posteriori* en sa faveur.
4. **Format mixte pour V3.2 aussi** (RoPE en BF16, latent en FP8). *Effet :* en BF16 pur, la
   baseline V3.2 grimperait à 70 KiB/token et le ratio à ~22× — donc au-delà de l'annonce. Le
   format mixte est le choix qui rend les deux modèles comparables.
5. **FLOPs des GEMM = 2 × paramètres activés.** Approximation standard ; ignore les normalisations,
   softmax et le surcoût mHC (que le papier chiffre à 6,7 % de temps mural).
6. **Matériel :** H800/H100 3,35 TB/s et 1 979 TFLOPS FP8 ; B200 8 TB/s et 4 500 TFLOPS FP8 ;
   MFU 40 % pour le prefill. Spécifications publiques, pas de mesure.

## 5. Limites méthodologiques majeures

**À prendre au sérieux avant de citer ces résultats.**

1. **Les cartes d'attention viennent de GPT-2** — un modèle de 117 M paramètres, contexte 1024,
   attention **dense**. Un modèle entraîné *nativement* creux sur 1 M de contexte a des
   distributions d'attention façonnées par la sparsité elle-même. Les **valeurs absolues** des
   expériences A2, A3 et C ne se transposent donc pas. Ce qui se transpose est la **structure**
   (concentration extrême, aiguilles isolées, absence de regroupement spatial), documentée comme
   universelle, et le **sens** des écarts entre conditions.
   **Atténuation partielle :** l'expérience E réplique les résultats structurels sur
   **SmolLM2-135M** (RoPE, GQA, 30 couches) — architecture et entraînement différents. Les quatre
   résultats tiennent, dans le même sens, avec des amplitudes qui varient (de +9,8 % à +6,1 % pour
   le gain à bande passante égale ; de 53,5 % à 45,2 % pour l'amortissement spéculatif). **Les
   amplitudes de ce rapport sont donc dépendantes du modèle ; les classements ne le sont pas.**
   Il reste que les deux modèles sont petits et à attention dense : la validation sur un modèle
   nativement creux à long contexte demeure nécessaire.
2. **Scores oracles.** Les expériences A2 et C sélectionnent les blocs par la somme des attentions
   *réelles*. Un indexeur réel est imparfait ; mes chiffres sont donc des **bornes supérieures**.
   L'expérience A3 est la seule à simuler explicitement le scoring par clé compressée.
3. **Métriques intermédiaires.** Je mesure de la masse d'attention récupérée, de l'erreur de logit
   et des KL — pas de la perplexité ni des scores de benchmark. Le lien entre ces proxys et la
   qualité finale est plausible mais non établi ici.
4. **Un seul modèle pour l'énergie RoPE** (SmolLM2-135M). Le papier Block-GTQ valide sur dix
   modèles ; ma vérification confirme sa prémisse mais n'établit pas la généralité du gradient de
   fréquence que je rapporte en plus (d'où Q4).
5. **Aucune mesure de temps réelle.** Pas de GPU sur la machine. Tous les temps sont des modèles
   analytiques à bande passante et MFU nominales — donc optimistes, sans contention, protocole ni
   effets de noyau.
6. **Le quantificateur de l'expérience B** est uniforme symétrique à échelle par (token, bloc),
   plus simple que TQ-MSE. Les gains uniforme→alloué rapportés sont donc probablement
   **conservateurs**.
7. **γ et α du décodage spéculatif** sont des paramètres libres, non mesurés sur un vrai couple
   brouillon/cible.

## 6. Hypothèses que j'ai testées et qui se sont révélées fausses

Signalées ici parce qu'elles sont instructives et qu'un lecteur pourrait les reformer.

- **« La compression par blocs marche parce que l'attention est spatialement groupée. »** Fausse :
  autocorrélation de 0,207 au décalage 1, 30,4 % d'aiguilles isolées. Le mécanisme réel est la
  préservation de la détectabilité par le score de somme, le coût étant une dilution de budget.
- **« La taille du KV cache interdit la désagrégation prefill/decode en contexte long. »** Fausse
  à prefill froid : le transfert ne pèse que 1,9 % à 1 M, et cette part *décroît* avec le
  contexte. Vraie seulement sous réutilisation de préfixe, où il n'y a plus de prefill à amortir.
- **« Le maximum est le meilleur compresseur pour préserver les aiguilles. »** Fausse : β→∞ est
  légèrement moins bon que β≈0,5, car le max ignore les blocs portant plusieurs tokens moyens.

## 7. Ce qui reste à vérifier en priorité

1. Refaire les expériences A2/A3/C sur un modèle **nativement creux** à long contexte
   (DeepSeek-V3.2-Exp est public) plutôt que sur GPT-2. C'est la limite la plus sérieuse.
2. Confirmer l'hypothèse du cache d'indexation en FP4 pour V4, en lisant l'implémentation publiée
   (`huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`, citée en note 1 du papier).
   Cela lèverait l'hypothèse la plus sensible du calcul de KV cache.
3. Valider le gradient d'énergie RoPE sur plusieurs modèles (Q4).
