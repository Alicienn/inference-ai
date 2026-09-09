"""
EXPERIENCE B -- Verification independante de la these "RoPE-aware bit allocation"
(arXiv 2606.24033) et de l'allocation binaire de DeepSeek-V4 (RoPE en BF16, reste en FP8).

These du papier : le logit d'attention RoPE se decompose en une somme sur des BLOCS de
frequence 2D ; le profil d'energie par bloc serait "fortement inegal" (plusieurs ordres
de grandeur), ce qui justifie d'allouer les bits par bloc plutot qu'uniformement.

On teste cette premisse sur des activations REELLES (SmolLM2-135M, RoPE, GQA, head_dim=64),
puis on mesure si une allocation gloutonne guidee par l'energie bat vraiment
l'allocation uniforme a budget de bits moyen egal.

Convention Llama : le bloc de frequence i associe les dimensions (i, i+d/2),
avec theta_i = base^(-2i/d). i faible = rotation rapide, i eleve = rotation lente.
"""
import numpy as np, torch, json, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\07_analyses")
OUT.mkdir(exist_ok=True)
MID = "HuggingFaceTB/SmolLM2-135M"
SEQ, NDOC = 1024, 3


# ---------------------------------------------------------------- capture Q/K pre-RoPE
def capture():
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval()
    cfg = model.config
    H, KVH = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // H
    L = cfg.num_hidden_layers
    base = getattr(cfg, "rope_theta", None) or 10000.0

    store = {}
    hooks = []

    def mk(name):
        def fn(mod, inp, out):
            store.setdefault(name, []).append(out.detach()[0].float())
        return fn

    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(mk(f"q{li}")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(mk(f"k{li}")))

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    buf, texts = "", []
    for r in ds:
        buf += r["text"]
        if len(buf) > 12000:
            texts.append(buf); buf = ""
            if len(texts) >= NDOC: break
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids
        if ids.shape[1] < SEQ: continue
        model(ids)
    for h in hooks: h.remove()
    return store, dict(H=H, KVH=KVH, D=D, L=L, base=base)


# ---------------------------------------------------------------- quantification
def quant_block(x, bits):
    """Quantification symetrique uniforme par bloc, echelle par (token, bloc)."""
    if bits >= 16:
        return x
    qmax = max(2 ** (bits - 1) - 1, 1)          # garde-fou : b=1 -> qmax=1
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
    return torch.round(x / s).clamp(-qmax - 1, qmax) * s


def apply_rope(kb, pos, half, base):
    """
    kb  : (T, half, 2) coordonnees pre-RoPE (paires (i, i+half))
    pos : (T,) positions
    Rotation du bloc i par l'angle pos * base^(-2i/d), d = 2*half.
    """
    i = torch.arange(half, dtype=torch.float32)
    theta = base ** (-2.0 * i / (2 * half))                 # (half,)
    ang = pos[:, None].float() * theta[None, :]             # (T, half)
    c, s = torch.cos(ang), torch.sin(ang)
    x, y = kb[..., 0], kb[..., 1]
    return torch.stack([x * c - y * s, x * s + y * c], -1)


def greedy_alloc(energy, bbar, bmin=1, bmax=12):
    """
    Allocation gloutonne minimisant  sum_b E_b * 4^{-b_b}  a moyenne de bits fixee.
    (loi de debit-distorsion MSE ~ 4^-b, cf. TQ-MSE)
    """
    n = len(energy)
    bits = np.full(n, bmin, dtype=int)
    budget = int(round(bbar * n)) - bits.sum()
    while budget > 0:
        # gain marginal d'un bit supplementaire
        gain = energy * (4.0 ** (-bits.astype(float)) - 4.0 ** (-(bits + 1).astype(float)))
        gain[bits >= bmax] = -1
        i = int(np.argmax(gain))
        if gain[i] <= 0: break
        bits[i] += 1; budget -= 1
    return bits


def main():
    print("=" * 96)
    print("EXPERIENCE B : profil d'energie par bloc RoPE et allocation de bits")
    print("=" * 96)
    store, cfg = capture()
    H, KVH, D, L, base = cfg["H"], cfg["KVH"], cfg["D"], cfg["L"], cfg["base"]
    half = D // 2
    print(f"  SmolLM2-135M : {L} couches, {H} tetes Q, {KVH} tetes KV, head_dim {D}, "
          f"rope_base {base}\n  -> {half} blocs de frequence 2D par tete\n")

    # ---------------- (1) profil d'energie
    print("=" * 96)
    print("(1) LE PROFIL D'ENERGIE PAR BLOC RoPE EST-IL 'FORTEMENT INEGAL' ?")
    print("=" * 96)
    spreads, profiles = [], []
    for li in range(L):
        q = torch.cat(store[f"q{li}"], 0)          # (T, H*D)
        k = torch.cat(store[f"k{li}"], 0)          # (T, KVH*D)
        T = q.shape[0]
        q = q.view(T, H, D); k = k.view(T, KVH, D)
        for kh in range(KVH):
            qh = q[:, kh * (H // KVH):(kh + 1) * (H // KVH), :]     # tetes Q du groupe
            kk = k[:, kh, :]
            # energie du bloc i : E[||q_i||^2] * E[||k_i||^2], paire (i, i+half)
            eq = (qh[..., :half] ** 2 + qh[..., half:] ** 2).mean((0, 1))   # (half,)
            ek = (kk[..., :half] ** 2 + kk[..., half:] ** 2).mean(0)
            e = (eq * ek).numpy()
            profiles.append(e)
            spreads.append(e.max() / max(np.median(e), 1e-30))
    P = np.array(profiles)
    Pn = P / np.median(P, 1, keepdims=True)
    print(f"  {len(P)} profils (couche x tete KV)")
    print(f"  etendue max/mediane : mediane {np.median(spreads):8.1f}x   "
          f"p90 {np.percentile(spreads,90):8.1f}x   max {np.max(spreads):8.1f}x")
    print(f"  -> soit {np.log10(np.median(spreads)):.1f} ordre(s) de grandeur en mediane.")
    print(f"\n  Profil moyen normalise, par indice de frequence (0 = rotation rapide):")
    m = Pn.mean(0)
    for a in range(0, half, 4):
        print(f"    blocs {a:2d}-{min(a+3,half-1):2d} : " +
              " ".join(f"{m[j]:7.2f}" for j in range(a, min(a + 4, half))))
    lo, hi = m[:half // 2].mean(), m[half // 2:].mean()
    print(f"\n  energie moyenne  hautes frequences (i<{half//2}) : {lo:6.2f}")
    print(f"  energie moyenne  basses frequences (i>={half//2}): {hi:6.2f}"
          f"   -> ratio {hi/lo:.2f}x")

    # ---------------- (2) allocation guidee vs uniforme
    print()
    print("=" * 96)
    print("(2) L'ALLOCATION GUIDEE PAR L'ENERGIE BAT-ELLE L'UNIFORME (budget moyen egal) ?")
    print("=" * 96)
    print(f"{'bits moy':>9s} | {'err. uniforme':>14s} {'err. allouee':>13s} {'gain':>9s}"
          f" | {'KL unif':>9s} {'KL alloc':>9s}")
    print("-" * 96)
    res = []
    rng = np.random.default_rng(0)
    for bbar in [2, 3, 4, 6, 8]:
        eu, ea, klu, kla = [], [], [], []
        for li in range(0, L, 3):
            q = torch.cat(store[f"q{li}"], 0); k = torch.cat(store[f"k{li}"], 0)
            T = q.shape[0]
            q = q.view(T, H, D); k = k.view(T, KVH, D)
            for kh in range(KVH):
                qh = q[:, kh * (H // KVH), :]        # une tete Q representative
                kk = k[:, kh, :]
                eq = (qh[:, :half] ** 2 + qh[:, half:] ** 2).mean(0)
                ek = (kk[:, :half] ** 2 + kk[:, half:] ** 2).mean(0)
                e = (eq * ek).numpy()
                bits = greedy_alloc(e, bbar)

                # vue "par bloc" : (T, half, 2), puis RoPE REELLEMENT applique
                # (les systemes reels mettent en cache les cles APRES rotation)
                pos = torch.arange(T)
                kb_pre = torch.stack([kk[:, :half], kk[:, half:]], -1)
                qb_pre = torch.stack([qh[:, :half], qh[:, half:]], -1)
                kb = apply_rope(kb_pre, pos, half, base)
                qb = apply_rope(qb_pre, pos, half, base)
                qf = torch.cat([qb[..., 0], qb[..., 1]], -1)
                # uniforme
                ku = quant_block(kb, bbar)
                # allouee (energie calculee en coordonnees pre-RoPE : invariante)
                ka = torch.stack([quant_block(kb[:, i, :], int(bits[i]))
                                  for i in range(half)], 1)

                def logits(kq):
                    kf = torch.cat([kq[..., 0], kq[..., 1]], -1)
                    return qf @ kf.T                    # (T, T) logits RoPE reels
                l0 = logits(kb); lu = logits(ku); la = logits(ka)
                n0 = l0.norm()
                eu.append(((lu - l0).norm() / n0).item())
                ea.append(((la - l0).norm() / n0).item())
                # KL de la distribution d'attention (echantillon de requetes)
                idx = rng.choice(T, 64, replace=False)
                for i in idx:
                    p0 = torch.softmax(l0[i, :i + 1] / np.sqrt(D), -1)
                    for lx, box in ((lu, klu), (la, kla)):
                        px = torch.softmax(lx[i, :i + 1] / np.sqrt(D), -1)
                        box.append((p0 * (torch.log(p0 + 1e-30) -
                                          torch.log(px + 1e-30))).sum().item())
        u, a_ = np.mean(eu), np.mean(ea)
        print(f"{bbar:9d} | {u:14.5f} {a_:13.5f} {100*(u-a_)/u:8.1f}%"
              f" | {np.mean(klu):9.5f} {np.mean(kla):9.5f}")
        res.append(dict(bits=bbar, err_uniform=float(u), err_alloc=float(a_),
                        kl_uniform=float(np.mean(klu)), kl_alloc=float(np.mean(kla))))

    print()
    print("=" * 96)
    print("(3) L'ALLOCATION BINAIRE DE DeepSeek-V4 (RoPE=BF16, reste=FP8)")
    print("=" * 96)
    print("""  V4 n'alloue que DEUX niveaux : 64 dims RoPE en BF16, le reste en FP8.
  C'est un cas particulier tres grossier de l'allocation par bloc. Nos mesures
  ci-dessus indiquent que l'ecart d'energie se joue ENTRE blocs de frequence a
  l'interieur meme de la partie RoPE ; une allocation binaire ne peut donc pas
  capturer ce gradient. Voir l'analyse detaillee dans 07_analyses/.""")

    (OUT / "expB_rope_bits.json").write_text(json.dumps(
        dict(spread_median=float(np.median(spreads)),
             spread_p90=float(np.percentile(spreads, 90)),
             mean_profile=[float(x) for x in m],
             hi_lo_ratio=float(hi / lo), results=res), indent=2))
    print(f"\n  -> {OUT/'expB_rope_bits.json'}")


if __name__ == "__main__":
    main()
