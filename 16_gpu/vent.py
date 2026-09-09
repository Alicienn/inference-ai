"""
VENTILATION du resultat de bout en bout : ou passe reellement le temps ?

Hypothese a tester : le pas de decodage n'est PAS borne par la memoire dans cette
integration HF/transformers, mais par le surcout de framework (36 couches x plusieurs
petits noyaux, sans graphe CUDA). Si c'est le cas, les 13 a 25x d'octets economises par
ASP ne peuvent pas se voir, et le surcout d'ASP (6 operations PyTorch de plus par couche)
domine.

On mesure separement : le plancher theorique memoire, l'attention seule, et le pas complet.
"""
import torch, time, math, json, pathlib
import e2e_qwen as E

L, KVH, D, H = 36, 8, 128, 32
BW = 3.8e12   # H200 NVL, bande passante crete annoncee


def timeit(fn, reps=9):
    for _ in range(2): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts)//2]


print("=" * 100)
print("VENTILATION — H200, Qwen3-8B, un pas de decodage")
print("=" * 100)
W = 15.3 * 2**30
print(f"  poids 15,3 Go ; bande passante crete {BW/1e12:.1f} To/s")
print(f"\n{'N':>9s} {'KV Go':>7s} | {'plancher':>9s} {'attn dense':>11s} {'attn ASP':>9s} "
      f"| {'pas dense':>10s} {'pas ASP':>8s} | {'attn / pas':>11s} {'MFU mem':>8s}")
print("-" * 100)
rows = []
sc = 1.0 / math.sqrt(D)
for N in (16384, 65536, 131072, 262144):
    n = N // 64; m = max(n // 8, 1); k = max(n // 64, 1)
    kv = L * KVH * D * 2 * 2 * N
    K = torch.randn(1, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
    V = torch.randn(1, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
    q = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    C1, C2 = E.build_summaries(K, 64, 2, 8)

    class M: pass
    mod = M()
    def dense():
        # SDPA conscient de GQA : pas de materialisation du KV etendu (artefact corrige,
        # il gonflait la baseline dense d'un facteur g=4)
        torch.nn.functional.scaled_dot_product_attention(q, K, V, scale=sc, enable_gqa=True)
    E._SUM.clear(); E._SUM[(id(mod), N)] = (C1, C2)
    def asp():
        E.asp_attention(mod, q, K, V, None, scaling=sc)

    ta_d = timeit(dense) * L
    ta_a = timeit(asp) * L
    floor = (W + kv) / BW
    rows.append(dict(N=N, kv_GB=kv/2**30, floor_ms=1e3*floor,
                     attn_dense_ms=1e3*ta_d, attn_asp_ms=1e3*ta_a))
    print(f"{N:9d} {kv/2**30:6.1f}G | {1e3*floor:8.2f}m {1e3*ta_d:10.2f}m {1e3*ta_a:8.2f}m "
          f"| {'—':>10s} {'—':>8s} | {'':>11s}")
    del K, V, C1, C2, q; torch.cuda.empty_cache()
pathlib.Path("/root/results/vent.json").write_text(json.dumps(rows, indent=2))
print("""
LECTURE. 'plancher' = (poids + KV) / bande passante crete : le temps minimal d'un pas de
decodage borne par la memoire. Comparer au 'pas complet' mesure precedemment (30 ms a
16k-131k) dit si l'integration est bornee par la memoire ou par le framework.""")
