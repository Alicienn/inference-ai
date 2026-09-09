"""
Modele analytique independant du cout d'inference de DeepSeek-V3.2 / V4-Flash / V4-Pro.

Objectif : reconstruire, a partir des SEULS hyperparametres publies dans les papiers,
  (a) le nombre de parametres totaux et actives  -> validation du modele
  (b) les FLOPs de decodage par token vs longueur de contexte
  (c) la taille du KV cache vs longueur de contexte
puis confronter aux affirmations de DeepSeek-V4 (arXiv 2606.19348, Fig. 1 et sec. 1) :
  V4-Flash @1M : 10% des FLOPs (9.8x) et 7% du KV cache (13.7x) de V3.2
  V4-Pro   @1M : 27% des FLOPs (3.7x) et 10% du KV cache (9.5x) de V3.2
  KV cache V4 ~= 2% d'une baseline BF16 GQA8 (head dim 128)

Sources hyperparametres :
  V3/V3.2 : arXiv 2412.19437 sec 4.2 ; arXiv 2512.02556 (DSA top-k=2048)
  V4      : arXiv 2606.19348 sec 4.2.1
"""
import numpy as np

GB = 1024 ** 3
KB = 1024

# ----------------------------------------------------------------------------
# Precisions (octets par element)
# ----------------------------------------------------------------------------
BF16, FP8, FP4 = 2.0, 1.0, 0.5


class Cfg:
    pass


# ---------------------------------------------------------------- V3.2 (=V3 + DSA)
v32 = Cfg()
v32.name = "DeepSeek-V3.2"
v32.L = 61
v32.d = 7168
v32.vocab = 128_000
# MLA
v32.n_h = 128        # tetes d'attention
v32.d_h = 128        # dim par tete
v32.d_c = 512        # dim compression KV (latent)
v32.d_cq = 1536      # dim compression query
v32.d_R = 64         # dim RoPE decouplee
# DSA
v32.topk = 2048
v32.n_I = 64         # tetes indexeur (config publique V3.2-Exp)
v32.c_I = 128        # dim tete indexeur
# MoE : 3 premieres couches denses, puis MoE
v32.dense_layers = 3
v32.n_routed = 256
v32.n_shared = 1
v32.n_act = 8
v32.d_ff_exp = 2048
v32.d_ff_dense = 18432   # FFN dense de V3

# ---------------------------------------------------------------- V4-Flash
f = Cfg()
f.name = "DeepSeek-V4-Flash"
f.L = 43
f.d = 4096
f.vocab = 128_000
f.n_swa_layers = 2       # 2 premieres couches : sliding window pur
f.m = 4                  # taux compression CSA
f.mp = 128               # taux compression HCA
f.topk = 512             # entrees compressees selectionnees (CSA)
f.n_I, f.c_I = 64, 128   # indexeur
f.n_h, f.c = 64, 512     # tetes de query, dim tete (= dim entree KV partagee)
f.d_cq = 1024            # dim compression query
f.g, f.d_g = 8, 1024     # projection de sortie groupee
f.n_win = 128
f.n_routed, f.n_shared, f.n_act = 256, 1, 6
f.d_ff_exp = 2048
f.hash_layers = 3        # routage Hash sur 3 premieres couches MoE (mais MoE partout)
f.n_hc = 4               # mHC

# ---------------------------------------------------------------- V4-Pro
p = Cfg()
p.name = "DeepSeek-V4-Pro"
p.L = 61
p.d = 7168
p.vocab = 128_000
p.n_swa_layers = 0       # 2 premieres couches = HCA (pas SWA pur)
p.m = 4
p.mp = 128
p.topk = 1024
p.n_I, p.c_I = 64, 128
p.n_h, p.c = 128, 512
p.d_cq = 1536
p.g, p.d_g = 16, 1024
p.n_win = 128
p.n_routed, p.n_shared, p.n_act = 384, 1, 6
p.d_ff_exp = 3072
p.n_hc = 4


# ============================================================================
# 1. COMPTAGE DE PARAMETRES  (validation du modele architectural)
# ============================================================================
def params_v32(c):
    """Parametres V3.2 : total et actives par token."""
    emb = c.vocab * c.d
    head = c.vocab * c.d

    # MLA par couche
    p_attn = 0
    p_attn += c.d * c.d_cq                                  # W_DQ
    p_attn += c.d_cq * c.n_h * (c.d_h + c.d_R)              # W_UQ (+ query RoPE decouplee)
    p_attn += c.d * (c.d_c + c.d_R)                         # W_DKV + k^R
    p_attn += c.d_c * c.n_h * c.d_h * 2                     # W_UK, W_UV
    p_attn += c.n_h * c.d_h * c.d                           # W_O
    # indexeur DSA
    p_idx = c.d * c.n_I * c.c_I                             # cles indexeur
    p_idx += c.d * c.d_cq * 0                               # queries partagent le latent
    p_idx += c.d_cq * c.n_I * c.c_I                         # W_IUQ
    p_idx += c.d * c.n_I                                    # poids par tete
    p_attn += p_idx

    ffn_dense = 3 * c.d * c.d_ff_dense
    exp = 3 * c.d * c.d_ff_exp
    moe_total = (c.n_routed + c.n_shared) * exp
    moe_act = (c.n_act + c.n_shared) * exp

    n_moe = c.L - c.dense_layers
    total = emb + head + c.L * p_attn + c.dense_layers * ffn_dense + n_moe * moe_total
    act = emb / c.vocab * 0 + c.L * p_attn + c.dense_layers * ffn_dense + n_moe * moe_act + head
    return total, act


def params_v4(c):
    """Parametres V4 (Flash/Pro) : total et actives par token."""
    emb = c.vocab * c.d
    head = c.vocab * c.d

    # --- projection de sortie groupee (commune CSA/HCA)
    #   n_h sorties de dim c -> g groupes -> chacun projete sur d_g -> concat -> d
    p_out = c.g * (c.c * c.n_h // c.g * c.d_g) + (c.d_g * c.g) * c.d

    # --- CSA
    p_csa = 0
    p_csa += 4 * c.d * c.c            # W_aKV, W_bKV, W_aZ, W_bZ
    p_csa += 2 * c.m * c.c            # biais positionnels B_a, B_b
    p_csa += c.d * c.d_cq             # W_DQ (latent partage query/indexeur)
    p_csa += c.d_cq * c.c * c.n_h     # W_UQ
    p_csa += c.d_cq * c.c_I * c.n_I   # W_IUQ
    p_csa += c.d * c.n_I              # W_w
    p_csa += 4 * c.d * c.c_I          # compression des cles indexeur
    p_csa += c.d * c.c                # branche sliding-window (KV non compresses)
    p_csa += p_out

    # --- HCA
    p_hca = 0
    p_hca += 2 * c.d * c.c            # W_KV, W_Z
    p_hca += c.mp * c.c               # biais B
    p_hca += c.d * c.d_cq             # W_DQ
    p_hca += c.d_cq * c.c * c.n_h     # W_UQ
    p_hca += c.d * c.c                # branche sliding-window
    p_hca += p_out

    # --- SWA pure (2 premieres couches de Flash) : approx = HCA sans compression
    p_swa = 2 * c.d * c.c + c.d * c.d_cq + c.d_cq * c.c * c.n_h + p_out

    n_csa, n_hca = layer_split(c)
    p_attn_total = n_csa * p_csa + n_hca * p_hca + c.n_swa_layers * p_swa

    # --- MoE (toutes les couches)
    exp = 3 * c.d * c.d_ff_exp
    moe_total = (c.n_routed + c.n_shared) * exp
    moe_act = (c.n_act + c.n_shared) * exp

    # --- mHC : matrices de melange n_hc x n_hc + transformations d'entree/sortie
    p_mhc = c.L * (c.n_hc * c.n_hc + 2 * c.n_hc * c.d)

    total = emb + head + p_attn_total + c.L * moe_total + p_mhc
    act = p_attn_total + c.L * moe_act + p_mhc + head
    return total, act


def layer_split(c):
    """Nombre de couches CSA / HCA (alternees apres les couches initiales)."""
    n_rest = c.L - c.n_swa_layers
    if c.name.endswith("Pro"):
        # 2 premieres = HCA, puis alternance sur les 59 restantes
        n_csa = (c.L - 2) // 2
        n_hca = c.L - 2 - n_csa + 2
        return n_csa, n_hca
    n_csa = n_rest // 2
    n_hca = n_rest - n_csa
    return n_csa, n_hca


# ============================================================================
# 2. KV CACHE (octets par token de contexte)
# ============================================================================
def kv_bytes_per_token_v32(c, idx_prec=FP8, mixed=True):
    """MLA : latent d_c + cle RoPE d_R, par couche. + cache cles indexeur."""
    if mixed:
        per_layer = c.d_c * FP8 + c.d_R * BF16
    else:
        per_layer = (c.d_c + c.d_R) * BF16
    idx = c.c_I * idx_prec               # cles indexeur, une par token
    return c.L * (per_layer + idx)


def kv_bytes_per_token_v4(c, idx_prec=FP4, n_csa=None, n_hca=None):
    """
    CSA  : 1 entree compressee tous les m tokens, dim c ; + cles indexeur compressees dim c_I
    HCA  : 1 entree compressee tous les m' tokens, dim c
    SWA  : fenetre bornee -> ne croit pas avec le contexte (compte a part)
    Format mixte : 64 dims RoPE en BF16, reste en FP8.
    """
    if n_csa is None:
        n_csa, n_hca = layer_split(c)
    ent = (c.c - 64) * FP8 + 64 * BF16          # une entree KV compressee
    idx_ent = max(c.c_I - 64, 0) * idx_prec + min(64, c.c_I) * idx_prec
    csa = ent / c.m + idx_ent / c.m
    hca = ent / c.mp
    return n_csa * csa + n_hca * hca


def kv_bytes_per_token_gqa8(L, kv_heads=8, head_dim=128, prec=BF16):
    return L * kv_heads * head_dim * 2 * prec   # K et V


# ============================================================================
# 3. FLOPs DE DECODAGE PAR TOKEN
# ============================================================================
def flops_v32(c, n, act_params):
    """FLOPs pour generer 1 token avec un contexte de n tokens."""
    mm = 2 * act_params                                   # tous les GEMM
    # attention centrale MLA absorbee, sur top-k entrees
    k = min(c.topk, n)
    qk = c.n_h * (c.d_c + c.d_R) * k * 2
    av = c.n_h * c.d_c * k * 2
    core = c.L * (qk + av)
    # indexeur : score contre TOUS les n tokens
    idx = c.L * (c.n_I * c.c_I * n * 2)
    return mm + core + idx, dict(mm=mm, core=core, idx=idx)


def flops_v4(c, n, act_params):
    n_csa, n_hca = layer_split(c)
    mm = 2 * act_params

    # --- CSA : attention sur topk entrees compressees + fenetre
    n_blocks = n / c.m
    k = min(c.topk, n_blocks) + c.n_win
    qk = c.n_h * c.c * k * 2
    av = c.n_h * c.c * k * 2
    core_csa = n_csa * (qk + av)
    # indexeur : score contre tous les n/m blocs compresses
    idx = n_csa * (c.n_I * c.c_I * n_blocks * 2)

    # --- HCA : attention dense sur n/m' entrees + fenetre
    kh = n / c.mp + c.n_win
    core_hca = n_hca * (c.n_h * c.c * kh * 2 * 2)

    # --- couches SWA pures
    core_swa = c.n_swa_layers * (c.n_h * c.c * c.n_win * 2 * 2)

    total = mm + core_csa + idx + core_hca + core_swa
    return total, dict(mm=mm, core=core_csa + core_hca + core_swa, idx=idx)


# ============================================================================
# EXECUTION
# ============================================================================
def fmt(x, u=1e9):
    return f"{x/u:8.1f}"


if __name__ == "__main__":
    print("=" * 78)
    print("1. VALIDATION : reconstruction des comptes de parametres")
    print("=" * 78)
    ref = {"DeepSeek-V3.2": (671e9, 37e9),
           "DeepSeek-V4-Flash": (284e9, 13e9),
           "DeepSeek-V4-Pro": (1600e9, 49e9)}
    res = {}
    for c, fn in ((v32, params_v32), (f, params_v4), (p, params_v4)):
        tot, act = fn(c)
        res[c.name] = (tot, act)
        rt, ra = ref[c.name]
        print(f"{c.name:20s} total {tot/1e9:7.1f}B (publie {rt/1e9:6.0f}B, "
              f"ecart {100*(tot-rt)/rt:+6.1f}%)   "
              f"actives {act/1e9:5.1f}B (publie {ra/1e9:4.0f}B, ecart {100*(act-ra)/ra:+6.1f}%)")

    # on utilise les valeurs PUBLIEES pour la suite (plus sures)
    act32, actF, actP = 37e9, 13e9, 49e9

    print()
    print("=" * 78)
    print("2. KV CACHE  (octets par token de contexte, et total a 1M)")
    print("=" * 78)
    kv32 = kv_bytes_per_token_v32(v32)
    kvF = kv_bytes_per_token_v4(f)
    kvP = kv_bytes_per_token_v4(p)
    ncF, nhF = layer_split(f)
    ncP, nhP = layer_split(p)
    print(f"repartition couches  Flash: {ncF} CSA / {nhF} HCA / {f.n_swa_layers} SWA"
          f"   Pro: {ncP} CSA / {nhP} HCA")
    for nm, kv in (("V3.2", kv32), ("V4-Flash", kvF), ("V4-Pro", kvP)):
        print(f"  {nm:10s} {kv/KB:8.2f} KiB/token   ->  {kv*1e6/GB:7.2f} GB @1M")
    print(f"\n  ratio KV  V3.2/V4-Flash = {kv32/kvF:5.2f}x  ({100*kvF/kv32:4.1f}%)"
          f"   [publie 13.7x / 7%]")
    print(f"  ratio KV  V3.2/V4-Pro   = {kv32/kvP:5.2f}x  ({100*kvP/kv32:4.1f}%)"
          f"   [publie  9.5x / 10%]")
    g8 = kv_bytes_per_token_gqa8(f.L)
    print(f"\n  baseline BF16 GQA8 (43 couches) : {g8/KB:.1f} KiB/token "
          f"-> V4-Flash = {100*kvF/g8:.2f}% [publie ~2%]")

    print()
    print("=" * 78)
    print("3. FLOPs DE DECODAGE PAR TOKEN vs CONTEXTE")
    print("=" * 78)
    print(f"{'contexte':>10s} | {'V3.2 (G)':>10s} {'Flash (G)':>10s} {'Pro (G)':>10s} |"
          f" {'x Flash':>8s} {'x Pro':>8s}")
    print("-" * 78)
    for n in (4_000, 32_000, 128_000, 256_000, 512_000, 1_000_000):
        t32, d32 = flops_v32(v32, n, act32)
        tF, dF = flops_v4(f, n, actF)
        tP, dP = flops_v4(p, n, actP)
        print(f"{n:10,d} | {t32/1e9:10.1f} {tF/1e9:10.1f} {tP/1e9:10.1f} |"
              f" {t32/tF:8.2f} {t32/tP:8.2f}")

    print()
    n = 1_000_000
    t32, d32 = flops_v32(v32, n, act32)
    tF, dF = flops_v4(f, n, actF)
    tP, dP = flops_v4(p, n, actP)
    print("  Decomposition @1M (GFLOPs) :")
    for nm, t, d in (("V3.2", t32, d32), ("V4-Flash", tF, dF), ("V4-Pro", tP, dP)):
        print(f"   {nm:10s} total {t/1e9:8.1f} = GEMM {d['mm']/1e9:7.1f}"
              f" + attn {d['core']/1e9:7.1f} + indexeur {d['idx']/1e9:8.1f}"
              f"   (indexeur = {100*d['idx']/t:4.1f}%)")
    print(f"\n  ratio FLOPs V3.2/V4-Flash = {t32/tF:5.2f}x ({100*tF/t32:4.1f}%)  [publie 9.8x / 10%]")
    print(f"  ratio FLOPs V3.2/V4-Pro   = {t32/tP:5.2f}x ({100*tP/t32:4.1f}%)  [publie 3.7x / 27%]")
