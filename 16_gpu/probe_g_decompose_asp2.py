"""
Test G -- pourquoi le pipeline ASP2 complet est plus lent qu'ASP1 malgre un update()
plus rapide (Test F). Decompose le cout par etape, comme Test B pour ASP1.
"""
import torch, time, math, json, pathlib
import e2e_qwen as E
import e2e_qwen_v2 as E2

OUT = pathlib.Path("/root/results/probe_g_decompose_asp2.json")
KVH, D, H = 8, 128, 32
s = 1.0 / math.sqrt(D)
g = H // KVH


def timeit(fn, reps=15):
    for _ in range(3):
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


print("=" * 100)
print("TEST G -- decomposition ASP2 par etape (comparaison directe avec ASP1, Test B)")
print("=" * 100)
hdr = (f"{'N':>9s} {'update':>8s} {'stage1':>8s} {'gatherS2':>9s} {'stage2':>8s} "
       f"{'gatherKV':>9s} {'attn':>8s} | {'somme ASP2':>11s} {'somme ASP1(*)':>14s}")
print(hdr)
print("-" * len(hdr))
for N in (16384, 65536, 131072, 262144):
    Lb, dprime = E2.CFG["Lb"], E2.CFG["dprime"]
    n = N // Lb
    m = max(n // E2.CFG["mfrac"], 1)
    k = max(n // E2.CFG["kfrac"], 1)
    K = torch.randn(1, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
    V = torch.randn(1, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
    q = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    qk = q.view(1, KVH, g, D)

    st = E2.IncrementalState(KVH, D, dprime, Lb, "cuda", torch.bfloat16)

    def t_update():
        return st.update(K)

    n_ready = t_update()

    def t_stage1():
        q_proj = torch.einsum('bhgd,hde->bhge', qk, st.basis)
        dots1 = torch.einsum('bhge,bhnle->bhgnl', q_proj, st.proj[:, :, :n_ready])
        s1 = dots1.max(-1).values.sum(2)
        return s1.topk(m, -1).indices

    surv = t_stage1()

    off = torch.arange(Lb, device="cuda")

    def t_gather_s2():
        idx_surv = (surv[..., None] * Lb + off).reshape(1, KVH, -1)
        return torch.gather(K, 2, idx_surv[..., None].expand(-1, -1, -1, D)).view(1, KVH, m, Lb, D)

    Ks_surv = t_gather_s2()

    def t_stage2():
        dots2 = torch.einsum('bhgd,bhmld->bhgml', qk, Ks_surv)
        s2 = dots2.max(-1).values.sum(2)
        return torch.gather(surv, 2, s2.topk(k, -1).indices)

    sel = t_stage2()

    def t_gather_kv():
        idx = (sel[..., None] * Lb + off).reshape(1, KVH, -1)
        tail = torch.arange(max(N - E2.CFG["nloc"], 0), N, device="cuda")
        sink = torch.arange(min(E2.CFG["nsink"], N), device="cuda")
        extra = torch.cat([sink, tail])[None, None, :].expand(1, KVH, -1)
        idx_full = torch.cat([idx, extra], -1).clamp_(0, N - 1)
        Ks = torch.gather(K, 2, idx_full[..., None].expand(-1, -1, -1, D))
        Vs = torch.gather(V, 2, idx_full[..., None].expand(-1, -1, -1, D))
        return Ks, Vs

    Ks, Vs = t_gather_kv()

    def t_attn():
        return torch.nn.functional.scaled_dot_product_attention(q, Ks, Vs, scale=s, enable_gqa=True)

    tu = timeit(t_update)
    t1 = timeit(t_stage1)
    tg1 = timeit(t_gather_s2)
    t2 = timeit(t_stage2)
    tg2 = timeit(t_gather_kv)
    ta = timeit(t_attn)
    somme2 = tu + t1 + tg1 + t2 + tg2 + ta
    print(f"{N:9d} {1e3*tu:7.3f}m {1e3*t1:7.3f}m {1e3*tg1:8.3f}m {1e3*t2:7.3f}m "
          f"{1e3*tg2:8.3f}m {1e3*ta:7.3f}m | {1e3*somme2:10.3f}m")
    del K, V, q, st, surv, Ks_surv, sel, Ks, Vs
    torch.cuda.empty_cache()

print("\n(*) voir 16_gpu/resultats/probe_b_decomposition.json pour la somme ASP1 equivalente.")
print("LECTURE. Si 'stage1' domine largement (contrairement a ASP1 ou c'etait 'update'/build),")
print("le cout s'est deplace : le score max necessite un tenseur 5D (n blocs x Lb cles x d'")
print("dims) touche a CHAQUE pas, meme quand les resumes sont deja construits -- plus cher")
print("par element que la reduction logsumexp compacte d'ASP1, malgre moins de donnees brutes.")
