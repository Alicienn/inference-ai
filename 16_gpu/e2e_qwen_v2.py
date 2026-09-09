"""
ASP v2 -- implementation du correctif valide par la recherche CPU (07-10 sept 2026) :

  1. Bug de vitesse trouve au Test B/D (16_gpu/probe_b_decomposition.py) : le code original
     (e2e_qwen.py) reconstruit TOUS les resumes de blocs a partir de zero a chaque pas de
     decodage -- O(N) par pas au lieu de O(Lb). Corrige ici par des TAMPONS PRE-ALLOUES a
     croissance amortie (doublement de capacite, jamais de torch.cat en boucle par pas) :
     seul le dernier bloc (partiel ou nouvellement complete) est retouche a chaque pas.

  2. Bug de qualite trouve par l'agent de recherche (20_sortie_attention, memoire
     technique_a5df3a0163da91a8671e et suivantes) : le score par MOYENNE de bloc est
     catastrophique aux deux etages (ecart de 2,2-2,4 nat vs un score par MAX), confirme
     independamment sur Qwen3-8B reel (21_rtx4090_reel/test_e_mean_vs_max_premier_etage.py,
     33 points d'ecart a k=1). Corrige ici par un score MAX aux deux etages :
       - etage 1 (grossier, n blocs) : max sur des cles projetees en d'=8 dims (base PCA
         causale, figee sur les 256 premiers tokens -- valide comme meilleure option).
       - etage 2 (fin, m survivants) : max sur les cles BRUTES (D dims) des blocs
         survivants, lues directement dans le cache K deja present -- pas de tampon
         separe necessaire, et valide comme dominant les resumes a octets egaux.

  PORTEE DE CE TEST : mesure UNIQUEMENT la vitesse (KV synthetique, comme e2e_qwen.py et
  tout le reste du projet). La qualite du score max / PCA a deja ete validee separement,
  sur activations reelles, en CPU. Ce n'est pas retteste ici.
"""
import torch, time, json, argparse, pathlib, math, gc, os

OUT = pathlib.Path("/root/results"); OUT.mkdir(exist_ok=True)
MODEL = os.environ.get("ASP_MODEL", "Qwen/Qwen3-8B")

CFG = dict(Lb=64, dprime=8, mfrac=8, kfrac=64, nloc=512, nsink=64, basis_n=256)
_STATE = {}


class IncrementalState:
    """Tampons a croissance amortie (doublement) : jamais de recopie O(N) par pas."""

    def __init__(self, KVH, D, dprime, Lb, device, dtype):
        self.KVH, self.D, self.dprime, self.Lb = KVH, D, dprime, Lb
        self.device, self.dtype = device, dtype
        self.basis = None                       # (KVH, D, dprime), fige apres le 1er fit
        self.cap = 64
        self.proj = torch.empty(1, KVH, self.cap, Lb, dprime, device=device, dtype=dtype)
        self.n_complete = 0                      # blocs finalises dans self.proj
        self.pending_start = 0                   # offset dans K du bloc courant (partiel)

    def _grow(self, need):
        if need <= self.cap:
            return
        newcap = max(need, self.cap * 2)
        newbuf = torch.empty(1, self.KVH, newcap, self.Lb, self.dprime,
                             device=self.device, dtype=self.dtype)
        newbuf[:, :, :self.cap] = self.proj
        self.proj, self.cap = newbuf, newcap

    def fit_basis(self, K0):
        """PCA causale figee, ajustee une seule fois sur les premiers tokens (K0)."""
        n = min(CFG["basis_n"], K0.shape[2])
        sample = K0[:, :, :n, :].float()                       # (1,KVH,n,D)
        basis = torch.empty(self.KVH, self.D, self.dprime, device=self.device)
        for h in range(self.KVH):
            x = sample[0, h] - sample[0, h].mean(0, keepdim=True)
            _, _, Vt = torch.linalg.svd(x, full_matrices=False)
            basis[h] = Vt[:self.dprime].T                      # (D, d')
        self.basis = basis.to(self.dtype)

    def update(self, K):
        """K : (1,KVH,N,D), cache complet actuel. Ne projette que les cles nouvelles."""
        if self.basis is None:
            self.fit_basis(K)
        N = K.shape[2]
        n_total = N // self.Lb
        if n_total > self.n_complete:
            self._grow(n_total)
            new_raw = K[:, :, self.n_complete * self.Lb: n_total * self.Lb]   # (1,KVH,x,D)
            nb = new_raw.shape[2] // self.Lb
            new_raw = new_raw.view(1, self.KVH, nb, self.Lb, self.D)
            new_proj = torch.einsum('bhnld,hde->bhnle', new_raw, self.basis)
            self.proj[:, :, self.n_complete:n_total] = new_proj
            self.n_complete = n_total
        return self.n_complete


def asp2_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    B, H, T, D = query.shape
    KVH, N = key.shape[1], key.shape[2]
    g = H // KVH
    sc = scaling if scaling is not None else 1.0 / math.sqrt(D)
    Lb, dprime = CFG["Lb"], CFG["dprime"]
    n = N // Lb
    if T > 1 or n < 64:
        o = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=(T > 1), scale=sc, enable_gqa=True)
        return o.transpose(1, 2).contiguous(), None

    ck = id(module)
    st = _STATE.get(ck)
    if st is None:
        st = IncrementalState(KVH, D, dprime, Lb, query.device, torch.bfloat16)
        _STATE[ck] = st
    n_ready = st.update(key)                     # O(nouveaux tokens), pas O(N)

    m = max(n_ready // CFG["mfrac"], 1)
    k = max(n_ready // CFG["kfrac"], 1)
    qk = query.view(B, KVH, g, D)

    # --- etage 1 : score MAX sur cles projetees d'=8 dims (remplace la moyenne) ---
    q_proj = torch.einsum('bhgd,hde->bhge', qk, st.basis)                       # (B,KVH,g,d')
    dots1 = torch.einsum('bhge,bhnle->bhgnl', q_proj, st.proj[:, :, :n_ready])  # (B,KVH,g,n,Lb)
    s1 = dots1.max(-1).values.sum(2)                                            # (B,KVH,n)
    surv = s1.topk(m, -1).indices                                               # (B,KVH,m)

    # --- etage 2 : score MAX sur cles BRUTES des survivants (remplace les resumes C2) ---
    off = torch.arange(Lb, device=query.device)
    idx_surv = (surv[..., None] * Lb + off).reshape(B, KVH, -1)                 # (B,KVH,m*Lb)
    Ks_surv = torch.gather(key, 2, idx_surv[..., None].expand(-1, -1, -1, D))
    Ks_surv = Ks_surv.view(B, KVH, m, Lb, D)
    dots2 = torch.einsum('bhgd,bhmld->bhgml', qk, Ks_surv)                      # (B,KVH,g,m,Lb)
    s2 = dots2.max(-1).values.sum(2)                                            # (B,KVH,m)
    sel = torch.gather(surv, 2, s2.topk(k, -1).indices)                         # (B,KVH,k)

    idx = (sel[..., None] * Lb + off).reshape(B, KVH, -1)
    tail = torch.arange(max(N - CFG["nloc"], 0), N, device=query.device)
    sink = torch.arange(min(CFG["nsink"], N), device=query.device)
    extra = torch.cat([sink, tail])[None, None, :].expand(B, KVH, -1)
    idx_full = torch.cat([idx, extra], -1).clamp_(0, N - 1)
    Ksel = torch.gather(key, 2, idx_full[..., None].expand(-1, -1, -1, D))
    Vsel = torch.gather(value, 2, idx_full[..., None].expand(-1, -1, -1, D))
    o = torch.nn.functional.scaled_dot_product_attention(query, Ksel, Vsel, scale=sc,
                                                         enable_gqa=True)
    return o.transpose(1, 2).contiguous(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+",
                    default=[16384, 32768, 65536, 131072, 262144, 524288])
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.cache_utils import DynamicCache
    import e2e_qwen as E
    ALL_ATTENTION_FUNCTIONS["asp"] = E.asp_attention
    ALL_ATTENTION_FUNCTIONS["asp2"] = asp2_attention

    cfg = AutoConfig.from_pretrained(MODEL)
    L, H, KVH = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    print(f"{MODEL} : {L} couches, {H} tetes Q, {KVH} KV, head_dim {D}, "
          f"max_pos {cfg.max_position_embeddings}")
    print(f"ASP v2 : Lb={CFG['Lb']}, d'={CFG['dprime']}, m=n/{CFG['mfrac']}, "
          f"k=n/{CFG['kfrac']}, base figee sur {CFG['basis_n']} tokens\n")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    w = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"poids charges : {w/2**30:.1f} Go\n")

    def make_cache(N):
        c = DynamicCache(config=model.config)
        for li in range(L):
            k = torch.randn(1, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            v = torch.randn(1, KVH, N, D, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            c.update(k, v, li)
        return c

    def step(cache, N):
        ids = torch.tensor([[1234]], device="cuda")
        pos = torch.tensor([[N]], device="cuda")
        with torch.no_grad():
            model(input_ids=ids, past_key_values=cache, position_ids=pos, use_cache=False)

    def timeit(fn):
        for _ in range(2): fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts)//2]

    rows = []
    hdr = (f"{'N tokens':>10s} {'n blocs':>8s} {'m':>6s} {'k':>4s} | {'KV Go':>7s} | "
           f"{'dense ms':>9s} {'ASP1 ms':>8s} {'ASP2 ms':>8s} | {'gain ASP1':>10s} {'gain ASP2':>10s}")
    print(hdr); print("-" * len(hdr))
    for N in a.Ns:
        n = N // CFG["Lb"]; m = max(n // CFG["mfrac"], 1); k = max(n // CFG["kfrac"], 1)
        kv = L * KVH * D * 2 * 2 * N
        free, _ = torch.cuda.mem_get_info()
        if kv > free * 0.5:
            print(f"{N:10d}  [saute : {kv/2**30:.0f} Go requis, {free/2**30:.0f} libres]")
            continue
        E._SUM.clear(); _STATE.clear()
        cache1 = make_cache(N)
        model.config._attn_implementation = "sdpa"
        for lay in model.model.layers: lay.self_attn.config._attn_implementation = "sdpa"
        td = timeit(lambda: step(cache1, N))
        del cache1; torch.cuda.empty_cache()

        cache2 = make_cache(N)
        model.config._attn_implementation = "asp"
        for lay in model.model.layers: lay.self_attn.config._attn_implementation = "asp"
        ta1 = timeit(lambda: step(cache2, N))
        del cache2; E._SUM.clear(); torch.cuda.empty_cache()

        cache3 = make_cache(N)
        _STATE.clear()
        model.config._attn_implementation = "asp2"
        for lay in model.model.layers: lay.self_attn.config._attn_implementation = "asp2"
        ta2 = timeit(lambda: step(cache3, N))
        del cache3; _STATE.clear(); torch.cuda.empty_cache()

        print(f"{N:10d} {n:8d} {m:6d} {k:4d} | {kv/2**30:6.1f}G | {1e3*td:8.2f} "
              f"{1e3*ta1:7.2f} {1e3*ta2:7.2f} | {td/ta1:9.2f}x {td/ta2:9.2f}x")
        rows.append(dict(N=N, n=n, m=m, k=k, kv_GB=kv/2**30, t_dense_ms=1e3*td,
                         t_asp1_ms=1e3*ta1, t_asp2_ms=1e3*ta2, gain_asp1=td/ta1, gain_asp2=td/ta2))
        gc.collect(); torch.cuda.empty_cache()
    (OUT / "e2e_qwen_v2.json").write_text(json.dumps(
        dict(model=MODEL, L=L, H=H, KVH=KVH, D=D, cfg=CFG, rows=rows), indent=2))
    print(f"\n-> {OUT/'e2e_qwen_v2.json'}")
    print("\nLECTURE. ASP1 = code original (bug de reconstruction complete + score moyenne).")
    print("ASP2 = correctif (tampons incrementaux + score max, cette session). Si gain_asp2 >")
    print("gain_asp1 nettement, le correctif ameliore reellement la vitesse, pas seulement")
    print("la qualite (deja validee separement en CPU).")


if __name__ == "__main__":
    main()
