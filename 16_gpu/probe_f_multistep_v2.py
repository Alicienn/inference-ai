"""
Test F -- reprise propre du Test D (probe_d_incremental.py), dont l'implementation etait
buguee (torch.cat en boucle -> reallocation a chaque pas, masquant le vrai gain).

Ici, IncrementalState (e2e_qwen_v2.py) utilise une croissance AMORTIE (doublement de
capacite, comme un tableau dynamique) : le cout de reallocation est rare, pas a chaque pas.

Simule une vraie boucle de decodage (le cache grandit de 1 token a chaque pas, sur
plusieurs dizaines de pas) et compare :
  - ASP1 (bug d'origine) : build_summaries recalcule integralement a chaque pas -- O(N).
  - ASP2 (corrige) : IncrementalState.update(), O(1) amorti par pas.
"""
import torch, time, json, pathlib
import e2e_qwen as E
import e2e_qwen_v2 as E2

OUT = pathlib.Path("/root/results/probe_f_multistep_v2.json")
Lb, r1, r2 = E.CFG["Lb"], E.CFG["r1"], E.CFG["r2"]


def timeit_total(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_case(N0, steps, KVH, D):
    K = torch.randn(1, KVH, N0, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)

    def asp1_loop():
        Ka = K.clone()
        for t in range(steps):
            E.build_summaries(Ka, Lb, r1, r2)
            newk = torch.randn(1, KVH, 1, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            Ka = torch.cat([Ka, newk], 2)

    def asp2_loop():
        Ka = K.clone()
        st = E2.IncrementalState(KVH, D, E2.CFG["dprime"], Lb, "cuda", torch.bfloat16)
        st.update(Ka)
        for t in range(steps):
            newk = torch.randn(1, KVH, 1, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            Ka = torch.cat([Ka, newk], 2)   # cout partage par les deux boucles (croissance du cache K lui-meme)
            st.update(Ka)

    reps = 3
    t1 = min(timeit_total(asp1_loop) for _ in range(reps))
    t2 = min(timeit_total(asp2_loop) for _ in range(reps))
    return t1, t2


def main():
    KVH, D = 8, 128
    steps = 128
    print("=" * 90)
    print(f"TEST F -- {steps} pas de decodage reels, cache qui grandit de 1 token/pas")
    print("=" * 90)
    hdr = f"{'N0':>9s} | {'ASP1 (s)':>10s} {'ASP2 (s)':>10s} | {'gain ASP2/ASP1':>15s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for N0 in (16384, 65536, 131072, 262144):
        free, _ = torch.cuda.mem_get_info()
        need = KVH * D * 2 * 2 * (N0 + steps) * 3
        if need > free * 0.6:
            print(f"{N0:9d} | [saute : VRAM insuffisante]")
            continue
        t1, t2 = run_case(N0, steps, KVH, D)
        print(f"{N0:9d} | {t1:10.4f} {t2:10.4f} | {t1/t2:15.2f}x")
        rows.append(dict(N0=N0, steps=steps, t_asp1_s=t1, t_asp2_s=t2, gain=t1/t2))
        torch.cuda.empty_cache()

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print("\nLECTURE. 'gain ASP2/ASP1' > 1 : le correctif incremental accelere reellement la")
    print("construction des resumes sur une vraie sequence de decodage. Attendu : le gain doit")
    print("CROITRE avec N0 (ASP1 est O(N) par pas, ASP2 est O(1) amorti par pas -- l'ecart")
    print("absolu grandit avec le contexte).")
    print("->", OUT)


if __name__ == "__main__":
    main()
