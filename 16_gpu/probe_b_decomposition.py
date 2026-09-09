"""
Test B -- decomposition du plateau ~24ms du selecteur ASP non fusionne.

Objectif : l'agent de recherche a identifie que le plateau observe (build_summaries + passe1
+ gather + passe2 + attention finale) domine l'attention dense jusqu'a N*=146k tokens sur
RTX4090, et propose plusieurs leviers pour l'abaisser (fusion du noyau, index 4 bits, moins
de candidats). Avant d'implementer quoi que ce soit, on decompose le plateau en ses etapes
pour savoir laquelle domine reellement -- sinon on optimise au hasard.

Etapes mesurees separement (evenements CUDA, mediane sur plusieurs repetitions) :
  1. build_summaries   : construction des resumes grossiers C1 et fins C2
  2. passe 1            : score logsumexp + topk sur tous les blocs -> survivants m
  3. gather C2s          : rassembler les resumes fins des survivants
  4. passe 2            : score logsumexp + topk sur les survivants -> k blocs retenus
  5. gather K/V finaux   : rassembler les cles/valeurs des blocs retenus + fenetre + puits
  6. attention finale    : SDPA sur le sous-ensemble
Le "lancement" (overhead fixe CUDA) est estime par extrapolation a N->0.
"""
import torch, time, math, json, pathlib
import e2e_qwen as E

OUT = pathlib.Path("/root/results/probe_b_decomposition.json")
L, KVH, D, H = 36, 8, 128, 32
s = 1.0 / math.sqrt(D)


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


def make_kv(N):
    K = torch.randn(1, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
    V = torch.randn(1, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
    q = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
    return K, V, q


rows = []
print("=" * 100)
print("TEST B -- decomposition du plateau du selecteur ASP (RTX4090)")
print("=" * 100)
hdr = (f"{'N':>9s} {'build':>8s} {'passe1':>8s} {'gatherC2':>9s} {'passe2':>8s} "
       f"{'gatherKV':>9s} {'attn':>8s} | {'somme':>8s} {'mesure e2e':>11s}")
print(hdr)
print("-" * len(hdr))
for N in (16384, 32768, 65536, 131072, 262144):
    Lb, r1, r2, mfrac, kfrac = E.CFG["Lb"], E.CFG["r1"], E.CFG["r2"], E.CFG["mfrac"], E.CFG["kfrac"]
    n = N // Lb
    m = max(n // mfrac, 1)
    k = max(n // kfrac, 1)
    K, V, q = make_kv(N)
    g = H // KVH
    qk = q.view(1, KVH, g, D)

    def t_build():
        return E.build_summaries(K, Lb, r1, r2)

    C1, C2 = t_build()

    def t_pass1():
        s1 = torch.logsumexp(torch.einsum('bhgd,bhnrd->bhgnr', qk, C1) * s, -1).sum(2)
        return s1.topk(m, -1).indices

    surv = t_pass1()

    def t_gather_c2():
        return torch.gather(C2, 2, surv[..., None, None].expand(-1, -1, -1, C2.shape[3], D))

    C2s = t_gather_c2()

    def t_pass2():
        s2 = torch.logsumexp(torch.einsum('bhgd,bhmrd->bhgmr', qk, C2s) * s, -1).sum(2)
        return torch.gather(surv, 2, s2.topk(k, -1).indices)

    sel = t_pass2()

    off = torch.arange(Lb, device="cuda")
    idx = (sel[..., None] * Lb + off).reshape(1, KVH, -1)
    tail = torch.arange(max(N - E.CFG["nloc"], 0), N, device="cuda")
    sink = torch.arange(min(E.CFG["nsink"], N), device="cuda")
    extra = torch.cat([sink, tail])[None, None, :].expand(1, KVH, -1)
    idx_full = torch.cat([idx, extra], -1).clamp_(0, N - 1)

    def t_gather_kv():
        Ks = torch.gather(K, 2, idx_full[..., None].expand(-1, -1, -1, D))
        Vs = torch.gather(V, 2, idx_full[..., None].expand(-1, -1, -1, D))
        return Ks, Vs

    Ks, Vs = t_gather_kv()

    def t_attn():
        return torch.nn.functional.scaled_dot_product_attention(q, Ks, Vs, scale=s, enable_gqa=True)

    def t_full():
        E._SUM.clear()
        class M: pass
        mod = M()
        E._SUM[(id(mod), N)] = (C1, C2)
        return E.asp_attention(mod, q, K, V, None, scaling=s)

    tb = timeit(t_build)
    t1 = timeit(t_pass1)
    tg1 = timeit(t_gather_c2)
    t2 = timeit(t_pass2)
    tg2 = timeit(t_gather_kv)
    ta = timeit(t_attn)
    te2e = timeit(t_full)
    somme = tb + t1 + tg1 + t2 + tg2 + ta
    print(f"{N:9d} {1e3*tb:7.3f}m {1e3*t1:7.3f}m {1e3*tg1:8.3f}m {1e3*t2:7.3f}m "
          f"{1e3*tg2:8.3f}m {1e3*ta:7.3f}m | {1e3*somme:7.3f}m {1e3*te2e:10.3f}m")
    rows.append(dict(N=N, build_ms=1e3*tb, pass1_ms=1e3*t1, gather_c2_ms=1e3*tg1,
                     pass2_ms=1e3*t2, gather_kv_ms=1e3*tg2, attn_ms=1e3*ta,
                     sum_ms=1e3*somme, e2e_measured_ms=1e3*te2e))
    del K, V, q, C1, C2, surv, C2s, sel, Ks, Vs
    torch.cuda.empty_cache()

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2))
print("\nLECTURE. 'somme' vs 'mesure e2e' : l'ecart est l'overhead de lancement de noyau non")
print("capture par la somme des etapes chronometrees individuellement (~10-20 lancements CUDA")
print("supplementaires pour la logique Python/PyTorch autour de chaque etape).")
print("La colonne la plus grande designe le vrai levier a optimiser en premier.")
print("->", OUT)
