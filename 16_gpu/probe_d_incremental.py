"""
Test D -- le vrai correctif du bug trouve au Test B, mesure sur une VRAIE sequence de pas.

Le Test B a montre que "build_summaries" (construire les resumes de blocs) est reconstruit
en entier a chaque pas de decodage (O(N)), alors que dans un decodage reel un seul token est
ajoute par pas : seul le DERNIER bloc (partiellement rempli) change, tous les autres blocs
sont finaux et leurs resumes ne doivent plus jamais etre recalcules.

Ce test :
  1. Implemente une version incrementale de build_summaries (O(Lb), independant de N).
  2. Simule une VRAIE boucle de decodage (le cache grandit de 1 token a chaque pas, sur
     plusieurs dizaines de pas) -- ce qu'aucun test precedent (e2e_qwen.py, vent.py) n'a
     fait : ils mesurent tous un seul pas a N fixe, jamais une sequence de pas.
  3. Compare trois strategies sur cette boucle reelle : dense, ASP actuel (reconstruction
     complete a chaque pas), ASP corrige (mise a jour incrementale).
"""
import torch, time, math, json, pathlib
import e2e_qwen as E

OUT = pathlib.Path("/root/results/probe_d_incremental.json")
Lb, r1, r2 = E.CFG["Lb"], E.CFG["r1"], E.CFG["r2"]
s = 1.0 / math.sqrt(128)


class IncrementalSummaries:
    """Maintient C1 (r1 segments/bloc) et C2 (r2 segments/bloc) pour un cache qui grandit
    token par token. Recalcule seulement le dernier bloc a chaque pas -- O(Lb), pas O(N)."""

    def __init__(self, KVH, D, r1, r2, Lb, device, dtype):
        self.KVH, self.D, self.r1, self.r2, self.Lb = KVH, D, r1, r2, Lb
        self.device, self.dtype = device, dtype
        self.C1 = torch.empty(1, KVH, 0, r1, D, device=device, dtype=dtype)
        self.C2 = torch.empty(1, KVH, 0, r2, D, device=device, dtype=dtype)
        self.n_complete = 0          # blocs finalises (jamais recalcules)

    def update(self, K):
        """K : (1,KVH,N,D) cache complet actuel. Ne retouche que le dernier bloc."""
        N = K.shape[2]
        n_total = N // self.Lb
        if n_total > self.n_complete:
            # de nouveaux blocs sont devenus complets depuis le dernier appel : les figer
            new_blocks = K[:, :, self.n_complete * self.Lb: n_total * self.Lb]
            nb = new_blocks.shape[2] // self.Lb
            nb_shaped = new_blocks.view(1, self.KVH, nb, self.Lb, self.D)
            c1_new = nb_shaped.view(1, self.KVH, nb, self.r1, self.Lb // self.r1, self.D).mean(4)
            c2_new = nb_shaped.view(1, self.KVH, nb, self.r2, self.Lb // self.r2, self.D).mean(4)
            self.C1 = torch.cat([self.C1, c1_new], 2)
            self.C2 = torch.cat([self.C2, c2_new], 2)
            self.n_complete = n_total
        # bloc courant (partiel), toujours recalcule -- cout O(Lb), independant de N
        rem = N - self.n_complete * self.Lb
        if rem == 0:
            return self.C1, self.C2
        tail = K[:, :, self.n_complete * self.Lb:]
        pad = self.Lb - rem
        if pad:
            tail = torch.cat([tail, tail[:, :, -1:].expand(-1, -1, pad, -1)], 2)
        tail = tail.view(1, self.KVH, 1, self.Lb, self.D)
        c1t = tail.view(1, self.KVH, 1, self.r1, self.Lb // self.r1, self.D).mean(4)
        c2t = tail.view(1, self.KVH, 1, self.r2, self.Lb // self.r2, self.D).mean(4)
        return torch.cat([self.C1, c1t], 2), torch.cat([self.C2, c2t], 2)


def timeit_total(fn, n):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_case(N0, steps, KVH, D, H):
    g = H // KVH
    K = torch.randn(1, KVH, N0, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
    V = torch.randn(1, KVH, N0, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)

    # --- Dense : SDPA sur tout le cache, a chaque pas ---
    def dense_loop():
        Kd, Vd = K.clone(), V.clone()
        for t in range(steps):
            q = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
            torch.nn.functional.scaled_dot_product_attention(q, Kd, Vd, scale=s, enable_gqa=True)
            newk = torch.randn(1, KVH, 1, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            Kd, Vd = torch.cat([Kd, newk], 2), torch.cat([Vd, newk], 2)

    # --- ASP actuel : reconstruction complete des resumes a chaque pas ---
    def asp_full_loop():
        Ka, Va = K.clone(), V.clone()
        for t in range(steps):
            q = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
            E.build_summaries(Ka, Lb, r1, r2)   # cout mesure au Test B : O(N), refait a chaque pas
            newk = torch.randn(1, KVH, 1, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            Ka, Va = torch.cat([Ka, newk], 2), torch.cat([Va, newk], 2)

    # --- ASP corrige : mise a jour incrementale ---
    def asp_incr_loop():
        Ka, Va = K.clone(), V.clone()
        inc = IncrementalSummaries(KVH, D, r1, r2, Lb, "cuda", torch.bfloat16)
        inc.update(Ka)   # etat initial
        for t in range(steps):
            q = torch.randn(1, H, 1, D, device="cuda", dtype=torch.bfloat16)
            newk = torch.randn(1, KVH, 1, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            Ka, Va = torch.cat([Ka, newk], 2), torch.cat([Va, newk], 2)
            inc.update(Ka)   # cout attendu : O(Lb), independant de N

    reps = 3
    td = min(timeit_total(dense_loop, 1) for _ in range(reps))
    tf = min(timeit_total(asp_full_loop, 1) for _ in range(reps))
    ti = min(timeit_total(asp_incr_loop, 1) for _ in range(reps))
    return td, tf, ti


def main():
    KVH, D, H = 8, 128, 32
    steps = 64
    print("=" * 100)
    print(f"TEST D -- {steps} pas de decodage REELS et CONSECUTIFS (le cache grandit de 1 a")
    print("chaque pas), construction des resumes seule (pas le reste du pipeline ASP)")
    print("=" * 100)
    hdr = f"{'N0':>9s} | {'dense (s)':>10s} {'ASP actuel (s)':>15s} {'ASP corrige (s)':>16s} | {'actuel/dense':>13s} {'corrige/dense':>14s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for N0 in (16384, 65536, 131072, 262144):
        free, _ = torch.cuda.mem_get_info()
        need = KVH * D * 2 * 2 * (N0 + steps) * 3  # marge Ka/Va + clones
        if need > free * 0.6:
            print(f"{N0:9d} | [saute : VRAM insuffisante]")
            continue
        td, tf, ti = run_case(N0, steps, KVH, D, H)
        print(f"{N0:9d} | {td:10.4f} {tf:15.4f} {ti:16.4f} | {tf/td:13.3f}x {ti/td:14.3f}x")
        rows.append(dict(N0=N0, steps=steps, t_dense_s=td, t_asp_full_s=tf, t_asp_incr_s=ti,
                         ratio_full=tf/td, ratio_incr=ti/td))
        torch.cuda.empty_cache()

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print("\nLECTURE. 'ASP actuel' = comportement du code tel qu'integre aujourd'hui (bug du Test B).")
    print("'ASP corrige' = avec mise a jour incrementale. Si 'corrige/dense' << 'actuel/dense',")
    print("le correctif suffit a rendre la construction des resumes negligeable face au dense.")
    print("Reserve : ne mesure QUE la construction des resumes sur une vraie boucle, pas le")
    print("reste du pipeline (tri, gather, attention finale) ni le forward complet du modele.")
    print("->", OUT)


if __name__ == "__main__":
    main()
