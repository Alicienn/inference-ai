# Ressources en ligne — Optimisation de l'inférence IA

Liens à donner au modèle de recherche pour suivre les nouveautés (au 2026-08-30).

## DeepSeek V4 / V4-Flash
- Model card officielle V4-Flash : https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash — MoE 284B (13B actifs), contexte 1M, orienté inférence agentique à bas coût.
- Variante DSpark : https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark
- API / annonces DeepSeek : https://api-docs.deepseek.com/ (voir les pages "news", ex. annonce V3.2-Exp : https://api-docs.deepseek.com/news/news250929/)
- Dépôt V3.2-Exp (DSA, kernels) : https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- Implémentation communautaire mHC : https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections
- Analyse de la lignée V3 → V3.2 (Sebastian Raschka) : https://magazine.sebastianraschka.com/p/technical-deepseek

## Moteurs d'inférence (documentation + blogs)
- vLLM : https://docs.vllm.ai — blog : https://vllm.ai/blog (ex. désagrégation prefill/decode mono-nœud : https://vllm.ai/blog/2026-04-07-moriio-kv-connector ; EAGLE-3 sur AMD Instinct : https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct)
- SGLang : https://docs.sglang.ai (RadixAttention, mode `--disaggregation-mode`)
- TensorRT-LLM : https://nvidia.github.io/TensorRT-LLM/
- LMDeploy : https://lmdeploy.readthedocs.io (désagrégation PD via DLSlime/Mooncake depuis v0.12)
- FlashInfer (kernels d'attention) : https://github.com/flashinfer-ai/flashinfer
- Mooncake (transfert/stockage KVCache) : https://github.com/kvcache-ai/mooncake
- Ray Serve, guide désagrégation PD : https://docs.ray.io/en/latest/serve/llm/user-guides/prefill-decode.html

## Veille / synthèses
- Sebastian Raschka — "LLM Research Papers: The 2026 List" : https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1
- Guide EAGLE-3 (E2E Networks) : https://www.e2enetworks.com/blog/Accelerating_LLM_Inference_with_EAGLE
- Architecture PD désagrégée chez Meta (vulgarisation) : https://jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode
- Déploiement V4-Flash (Spheron) : https://www.spheron.network/blog/deploy-deepseek-v4-flash-gpu-cloud/

## Sujets à surveiller (état août 2026)
- **Attention creuse native** (DSA → CSA/HCA de V4, LSA) : la voie dominante pour le contexte ≥1M tokens.
- **Quantization FP8/FP4 bout-en-bout** (poids + KV cache + calcul) sur Hopper/Blackwell.
- **Désagrégation prefill/decode** et transfert KV inter-nœuds : en cours de standardisation dans vLLM/SGLang/Dynamo.
- **Décodage spéculatif** : EAGLE-3 en production ; attention aux attaques type "Mistletoe" (arXiv 2605.14005) sur les déploiements partagés.
- **Routage/cascade multi-modèles** pour réduire le coût par requête (survey arXiv 2603.04445).
