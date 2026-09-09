"""
TEST FINAL -- le pipeline ASP complet et corrige (e2e_qwen_v2.asp2_attention, qui integre
DEJA les deux correctifs : tampons incrementaux + score max aux deux etages), mesure sur
une VRAIE boucle de decodage multi-pas dans le modele reel Qwen3-8B -- pas un instantane
a N fixe (comme tous les tests precedents e2e_qwen.py/vent.py), pas une brique isolee
(comme Test B/D/F), mais un vrai `model()` appele en boucle avec use_cache=True, le cache
grandissant naturellement token par token comme en generation reelle.

C'est le test qui manquait pour repondre a la question : le correctif, integre dans le
pipeline complet, rend-il vraiment ASP plus rapide que dense sur un vrai decodage ?

  - dense : sdpa natif
  - asp1  : pipeline original (bug : reconstruction complete + score moyenne)
  - asp2  : pipeline corrige (tampons incrementaux + score max) -- deja fonctionnel

Session GPU finale : ce script ne relance rien apres, les resultats sont ecrits sur disque
et rapatries immediatement.
"""
import torch, time, json, pathlib, gc, os

OUT = pathlib.Path("/root/results/e2e_multistep_final.json")
MODEL = os.environ.get("ASP_MODEL", "Qwen/Qwen3-8B")


def run_loop(model, L, impl, N0, steps, device="cuda"):
    from transformers.cache_utils import DynamicCache
    KVH = model.config.num_key_value_heads
    D = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    cache = DynamicCache(config=model.config)
    for li in range(L):
        k = torch.randn(1, KVH, N0, D, device=device, dtype=torch.bfloat16).mul_(0.05)
        v = torch.randn(1, KVH, N0, D, device=device, dtype=torch.bfloat16).mul_(0.05)
        cache.update(k, v, li)

    model.config._attn_implementation = impl
    for lay in model.model.layers:
        lay.self_attn.config._attn_implementation = impl

    ids = torch.tensor([[1234]], device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(steps):
            pos = torch.tensor([[N0 + t]], device=device)
            model(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=True)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    del cache
    gc.collect(); torch.cuda.empty_cache()
    return total


def main():
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import e2e_qwen as E
    import e2e_qwen_v2 as E2
    ALL_ATTENTION_FUNCTIONS["asp"] = E.asp_attention
    ALL_ATTENTION_FUNCTIONS["asp2"] = E2.asp2_attention

    cfg = AutoConfig.from_pretrained(MODEL)
    L, H, KVH = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    print(f"{MODEL} : {L} couches, {H} tetes Q, {KVH} KV, head_dim {D}\n")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    print(f"poids charges\n")

    steps = 32
    print("=" * 100)
    print(f"TEST FINAL -- {steps} pas de decodage REELS (model() en boucle, use_cache=True,")
    print("le cache grandit naturellement), pipeline ASP complet corrige")
    print("=" * 100)
    hdr = f"{'N0':>9s} | {'dense (s)':>10s} {'asp1 (s)':>10s} {'asp2 (s)':>10s} | {'gain asp1':>10s} {'gain asp2':>10s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for N0 in (16384, 65536, 131072, 262144):
        free, _ = torch.cuda.mem_get_info()
        kv_est = L * KVH * D * 2 * 2 * (N0 + steps)
        if kv_est > free * 0.55:
            print(f"{N0:9d} | [saute : VRAM insuffisante]")
            continue
        E._SUM.clear(); E2._STATE.clear()
        td = run_loop(model, L, "sdpa", N0, steps)
        E._SUM.clear(); E2._STATE.clear()
        t1 = run_loop(model, L, "asp", N0, steps)
        E._SUM.clear(); E2._STATE.clear()
        t2 = run_loop(model, L, "asp2", N0, steps)
        E._SUM.clear(); E2._STATE.clear()
        print(f"{N0:9d} | {td:10.4f} {t1:10.4f} {t2:10.4f} | {td/t1:10.3f}x {td/t2:10.3f}x")
        rows.append(dict(N0=N0, steps=steps, t_dense_s=td, t_asp1_s=t1, t_asp2_s=t2,
                         gain_asp1=td/t1, gain_asp2=td/t2))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(model=MODEL, steps=steps, rows=rows), indent=2))
    print("\nLECTURE. C'est le test decisif : un vrai model() en boucle, cache reel qui")
    print("grandit, pipeline ASP complet (pas une brique isolee). Si gain_asp2 > 1, le")
    print("correctif rend vraiment le decodage plus rapide que dense en pratique.")
    print("->", OUT)


if __name__ == "__main__":
    main()
