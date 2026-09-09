"""
Test C -- le cout du selecteur ASP est-il partage entre requetes d'un batch ?

Idee testee : sur GPU, une meme instruction s'execute en parallele sur tout le batch (SIMD).
Si le plateau ~24ms du selecteur est domine par du travail massivement parallele (et non par
des lancements de noyau sequentiels), agrandir le batch ne devrait presque rien couter en
temps supplementaire -- ce qui abaisserait le seuil de croisement N* d'un facteur ~batch,
sans changer une ligne de l'algorithme.

Protocole : cache KV independant par sequence du batch (pas de prefixe partage), meme N pour
toutes. On mesure dense et ASP a N fixe pour B in {1,2,4,8,16}, et on regarde si le temps par
sequence (temps_total / B) chute avec B.
"""
import torch, time, json, pathlib, math, gc

OUT = pathlib.Path("/root/results/probe_c_batch.json")
MODEL = "Qwen/Qwen3-8B"


def timeit(fn, reps=9):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.cache_utils import DynamicCache
    import e2e_qwen as E
    ALL_ATTENTION_FUNCTIONS["asp"] = E.asp_attention

    cfg = AutoConfig.from_pretrained(MODEL)
    Lc, H, KVH = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    print(f"{MODEL} charge : {Lc} couches, {H} tetes Q, {KVH} KV, head_dim {D}\n")

    def make_cache(N, B):
        c = DynamicCache(config=model.config)
        for li in range(Lc):
            k = torch.randn(B, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            v = torch.randn(B, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            c.update(k, v, li)
        return c

    def step(cache, N, B):
        ids = torch.full((B, 1), 1234, device="cuda", dtype=torch.long)
        pos = torch.full((B, 1), N, device="cuda", dtype=torch.long)
        with torch.no_grad():
            model(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=False)

    rows = []
    N = 65536  # taille fixe representative, choisie sous N* pour voir si le batch le fait bouger
    print(f"N={N} fixe -- balayage du batch B")
    hdr = f"{'B':>4s} | {'dense tot':>10s} {'ASP tot':>9s} | {'dense/seq':>10s} {'ASP/seq':>9s} | {'gain/seq':>9s}"
    print(hdr); print("-" * len(hdr))
    for B in (1, 2, 4, 8, 16, 32):
        n = N // E.CFG["Lb"]
        sm = Lc * KVH * n * (E.CFG["r1"] + E.CFG["r2"]) * D * 2 * B
        kv = Lc * KVH * D * 2 * 2 * N * B
        free, _ = torch.cuda.mem_get_info()
        if kv + sm > free * 0.75:
            print(f"{B:4d} | [saute : VRAM insuffisante]")
            continue
        E._SUM.clear()
        cache = make_cache(N, B)
        model.config._attn_implementation = "sdpa"
        for lay in model.model.layers:
            lay.self_attn.config._attn_implementation = "sdpa"
        td = timeit(lambda: step(cache, N, B))
        model.config._attn_implementation = "asp"
        for lay in model.model.layers:
            lay.self_attn.config._attn_implementation = "asp"
        ta = timeit(lambda: step(cache, N, B))
        print(f"{B:4d} | {1e3*td:9.2f}m {1e3*ta:8.2f}m | {1e3*td/B:9.3f}m {1e3*ta/B:8.3f}m "
              f"| {(td/B)/(ta/B):8.3f}x")
        rows.append(dict(B=B, N=N, t_dense_tot_ms=1e3*td, t_asp_tot_ms=1e3*ta,
                         t_dense_per_seq_ms=1e3*td/B, t_asp_per_seq_ms=1e3*ta/B))
        del cache; E._SUM.clear(); gc.collect(); torch.cuda.empty_cache()

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(dict(model=MODEL, N=N, rows=rows), indent=2))
    print("\nLECTURE. Si 'ASP/seq' baisse fortement quand B augmente (alors que 'dense/seq' reste")
    print("stable), le plateau du selecteur est bien parallelisable gratuitement par le GPU --")
    print("et batcher les requetes abaisse N* dans les memes proportions, sans changer le code.")
    print("Si 'ASP/seq' ne baisse pas, le plateau est deja sequentiel (lancements de noyau) et")
    print("le batching n'aide pas : seule la fusion (Test B) peut le reduire.")
    print("->", OUT)


if __name__ == "__main__":
    main()
