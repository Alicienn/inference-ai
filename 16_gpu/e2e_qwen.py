"""
VOLET 2 — ASP dans le chemin d'inference de Qwen3-8B, mesure de bout en bout (H200).

INTEGRATION. transformers 5.16 resout l'attention par
`ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, ...)`. On y enregistre
une implementation "asp" : c'est le point d'extension prevu, pas un monkeypatch. Le reste
du modele (36 couches, RMSNorm, MLP, RoPE, poids reels) est inchange et s'execute
normalement — la mesure porte donc sur un PAS DE DECODAGE COMPLET, pas sur l'attention
isolee.

CE QUI EST MESURE
  - `dense` : implementation sdpa native, attention sur tout le cache
  - `asp`   : passe grossiere sur les n blocs -> top-m -> passe fine -> top-k blocs,
              puis attention sur les blocs retenus + fenetre locale + puits

QUATRE LIMITES ASSUMEES, ecrites ici pour ne pas etre decouvertes apres coup :

1. `max_position_embeddings = 40960` pour Qwen3-8B. Au-dela de 40k on sort du contexte
   d'entrainement (il faudrait YaRN). La LATENCE et le MOTIF MEMOIRE restent identiques —
   c'est ce qu'on mesure — mais les sorties n'ont plus de sens linguistique. **On mesure
   de la performance, pas de la qualite.** La qualite de la selection a ete mesuree
   separement sur attention reelle (12_poc/).
2. Le KV cache est rempli de valeurs synthetiques a la bonne forme. Un prefill reel de
   500k tokens coute plusieurs minutes de calcul quadratique et ne changerait pas la
   latence de decodage, qui ne depend que de la forme du cache.
3. Les resumes de blocs sont des MOYENNES DE SEGMENTS (partition du bloc en r segments
   contigus, moyenne de chacun), un coreset pauvre mais reel et calculable en O(n). Le
   POC utilisait un k-centre, meilleur ; la latence ne depend que de leur TAILLE.
4. C'est une integration dans un modele reel, PAS dans un moteur de serving (vLLM/SGLang).
   Pas de batching continu, pas de KV pagine. Declare comme tel.
"""
import torch, time, json, argparse, pathlib, math, gc, os

OUT = pathlib.Path("/root/results"); OUT.mkdir(exist_ok=True)
MODEL = os.environ.get("ASP_MODEL", "Qwen/Qwen3-8B")

CFG = dict(Lb=64, r1=2, r2=8, mfrac=8, kfrac=64, nloc=512, nsink=64)
_SUM = {}          # cache des resumes : (id(module), N) -> (C1, C2)


def build_summaries(K, Lb, r1, r2):
    """Resumes par moyennes de segments. K:(B,KVH,N,D) -> C:(B,KVH,n,r,D)."""
    B, KVH, N, D = K.shape
    n = N // Lb
    Kb = K[:, :, :n * Lb].view(B, KVH, n, Lb, D)
    def seg(r):
        return Kb.view(B, KVH, n, r, Lb // r, D).mean(4)
    return seg(r1), seg(r2)


def asp_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    """Implementation d'attention ASP, signature de ALL_ATTENTION_FUNCTIONS."""
    B, H, T, D = query.shape
    KVH, N = key.shape[1], key.shape[2]
    g = H // KVH
    sc = scaling if scaling is not None else 1.0 / math.sqrt(D)
    Lb = CFG["Lb"]; n = N // Lb
    # prefill ou contexte trop court : on retombe sur l'attention dense
    if T > 1 or n < 64:
        o = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=(T > 1), scale=sc, enable_gqa=True)
        return o.transpose(1, 2).contiguous(), None

    m = max(n // CFG["mfrac"], 1); k = max(n // CFG["kfrac"], 1)
    ck = (id(module), N)
    if ck not in _SUM:
        _SUM.clear()
        _SUM[ck] = build_summaries(key, Lb, CFG["r1"], CFG["r2"])
    C1, C2 = _SUM[ck]

    qk = query.view(B, KVH, g, D)
    # passe 1 : grossiere, sur tous les blocs
    s1 = torch.logsumexp(torch.einsum('bhgd,bhnrd->bhgnr', qk, C1) * sc, -1).sum(2)
    surv = s1.topk(m, -1).indices                                        # (B,KVH,m)
    # passe 2 : fine, sur les survivants seulement
    C2s = torch.gather(C2, 2, surv[..., None, None].expand(-1, -1, -1, C2.shape[3], D))
    s2 = torch.logsumexp(torch.einsum('bhgd,bhmrd->bhgmr', qk, C2s) * sc, -1).sum(2)
    sel = torch.gather(surv, 2, s2.topk(k, -1).indices)                  # (B,KVH,k)

    off = torch.arange(Lb, device=query.device)
    idx = (sel[..., None] * Lb + off).reshape(B, KVH, -1)
    tail = torch.arange(max(N - CFG["nloc"], 0), N, device=query.device)
    sink = torch.arange(min(CFG["nsink"], N), device=query.device)
    extra = torch.cat([sink, tail])[None, None, :].expand(B, KVH, -1)
    idx = torch.cat([idx, extra], -1).clamp_(0, N - 1)
    Ks = torch.gather(key, 2, idx[..., None].expand(-1, -1, -1, D))
    Vs = torch.gather(value, 2, idx[..., None].expand(-1, -1, -1, D))
    # ARTEFACT CORRIGE : ne PAS materialiser le KV etendu par repeat_interleave, qui
    # gonfle artificiellement la baseline dense d'un facteur g=4. SDPA sait faire GQA.
    o = torch.nn.functional.scaled_dot_product_attention(query, Ks, Vs, scale=sc,
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
    ALL_ATTENTION_FUNCTIONS["asp"] = asp_attention

    cfg = AutoConfig.from_pretrained(MODEL)
    L, H, KVH = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    print(f"{MODEL} : {L} couches, {H} tetes Q, {KVH} KV, head_dim {D}, "
          f"max_pos {cfg.max_position_embeddings}")
    print(f"KV bf16 : {L*KVH*D*2*2/1024:.0f} Ko/token | "
          f"ASP : blocs {CFG['Lb']}, r1={CFG['r1']}, r2={CFG['r2']}, "
          f"m=n/{CFG['mfrac']}, k=n/{CFG['kfrac']}\n")

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
           f"{'dense ms':>9s} {'ASP ms':>8s} {'GAIN':>7s} | {'lu dense':>9s} "
           f"{'lu ASP':>8s} {'ratio':>7s}")
    print(hdr); print("-" * len(hdr))
    for N in a.Ns:
        n = N // CFG["Lb"]; m = max(n // CFG["mfrac"], 1); k = max(n // CFG["kfrac"], 1)
        kv = L * KVH * D * 2 * 2 * N
        free, _ = torch.cuda.mem_get_info()
        sm = L * KVH * n * (CFG["r1"] + CFG["r2"]) * D * 2
        if kv + sm > free * 0.75:
            print(f"{N:10d}  [saute : {(kv+sm)/2**30:.0f} Go requis, {free/2**30:.0f} libres]")
            continue
        _SUM.clear()
        cache = make_cache(N)
        model.config._attn_implementation = "sdpa"
        for lay in model.model.layers: lay.self_attn.config._attn_implementation = "sdpa"
        td = timeit(lambda: step(cache, N))
        model.config._attn_implementation = "asp"
        for lay in model.model.layers: lay.self_attn.config._attn_implementation = "asp"
        ta = timeit(lambda: step(cache, N))
        rd = L * KVH * N * D * 2 * 2
        ra = L * (KVH*n*CFG["r1"]*D*2 + KVH*m*CFG["r2"]*D*2
                  + KVH*(k*CFG["Lb"]+CFG["nloc"]+CFG["nsink"])*D*2*2)
        print(f"{N:10d} {n:8d} {m:6d} {k:4d} | {kv/2**30:6.1f}G | {1e3*td:8.2f} "
              f"{1e3*ta:7.2f} {td/ta:6.2f}x | {rd/1e6:8.1f}M {ra/1e6:7.1f}M {rd/ra:6.2f}x")
        rows.append(dict(N=N, n=n, m=m, k=k, kv_GB=kv/2**30, t_dense_ms=1e3*td,
                         t_asp_ms=1e3*ta, gain=td/ta, bytes_dense=rd, bytes_asp=ra,
                         byte_ratio=rd/ra))
        del cache; _SUM.clear(); gc.collect(); torch.cuda.empty_cache()
    (OUT / "e2e_qwen.json").write_text(json.dumps(
        dict(model=MODEL, L=L, H=H, KVH=KVH, D=D, cfg=CFG, rows=rows), indent=2))
    print(f"\n-> {OUT/'e2e_qwen.json'}")


if __name__ == "__main__":
    main()
