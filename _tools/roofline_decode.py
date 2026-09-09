"""
Analyse roofline du decodage : ce qui limite reellement la latence par token.

Idee centrale testee ici : pour l'attention creuse, il faut distinguer
  - KV_resident : empreinte memoire du cache (limite le batch x contexte)
  - KV_streamed : octets REELLEMENT lus a chaque token decode (limite la latence)
Pour l'attention dense ces deux quantites sont egales. L'attention creuse les
decouple, et c'est de ce decouplage que vient l'essentiel du gain de latence.

Depend de model_inference_cost.py pour les configurations.
"""
import numpy as np
from model_inference_cost import (v32, f, p, layer_split, flops_v32, flops_v4,
                                  kv_bytes_per_token_v32, kv_bytes_per_token_v4,
                                  BF16, FP8, FP4, GB, KB)

# ---------------------------------------------------------------- materiel
HW = {
    #          HBM GB, BW TB/s, FP8 TFLOPS dense
    "H800":  dict(mem=80,  bw=3.35e12, fp8=1979e12),
    "H200":  dict(mem=141, bw=4.80e12, fp8=1979e12),
    "B200":  dict(mem=192, bw=8.00e12, fp8=4500e12),
}


# ============================================================================
# Octets lus par token decode (par sequence)
# ============================================================================
def streamed_v32(c, n, idx_prec=FP8):
    """
    V3.2 / DSA : l'indexeur doit scorer TOUS les n tokens -> il lit tout le
    cache de cles indexeur. L'attention centrale ne lit que le top-k.
    """
    idx = c.L * n * c.c_I * idx_prec
    k = min(c.topk, n)
    core = c.L * k * (c.d_c * FP8 + c.d_R * BF16)
    return idx + core, dict(idx=idx, core=core)


def streamed_v4(c, n, idx_prec=FP4):
    """
    CSA : l'indexeur lit les n/m cles compressees, l'attention lit top-k entrees.
    HCA : PAS d'indexeur, mais attention dense -> lit les n/m' entrees.
    """
    n_csa, n_hca = layer_split(c)
    ent = (c.c - 64) * FP8 + 64 * BF16
    idx_ent = c.c_I * idx_prec

    nb = n / c.m
    idx = n_csa * nb * idx_ent
    core_csa = n_csa * (min(c.topk, nb) + c.n_win) * ent
    core_hca = n_hca * (n / c.mp + c.n_win) * ent
    return idx + core_csa + core_hca, dict(idx=idx, core=core_csa + core_hca)


# ============================================================================
# Poids : octets a lire (partages sur le batch)
# ============================================================================
def weight_bytes(name):
    """Routed experts en FP4 pour V4, FP8 sinon."""
    if name == "DeepSeek-V3.2":
        return 671e9 * FP8
    if name == "DeepSeek-V4-Flash":
        routed = 43 * 256 * 3 * 4096 * 2048
        return routed * FP4 + (284e9 - routed) * FP8
    routed = 61 * 384 * 3 * 7168 * 3072
    return routed * FP4 + (1600e9 - routed) * FP8


def expected_experts_read(n_routed, n_act, B):
    """Experts distincts touches par un batch de B tokens (approx binomiale)."""
    return n_routed * (1 - (1 - n_act / n_routed) ** B)


# ============================================================================
def analyse(n=1_000_000, hw="H800"):
    h = HW[hw]
    print("=" * 96)
    print(f"ROOFLINE DECODAGE  --  contexte {n:,} tokens  --  {hw} "
          f"(BW {h['bw']/1e12:.2f} TB/s, {h['mem']} GB, FP8 {h['fp8']/1e12:.0f} TFLOPS)")
    print("=" * 96)

    rows = []
    for c, kvfn, stfn, act in ((v32, kv_bytes_per_token_v32, streamed_v32, 37e9),
                               (f, kv_bytes_per_token_v4, streamed_v4, 13e9),
                               (p, kv_bytes_per_token_v4, streamed_v4, 49e9)):
        kv = kvfn(c) * n
        st, d = stfn(c, n)
        fl = (flops_v32 if c.name == "DeepSeek-V3.2" else flops_v4)(c, n, act)[0]
        rows.append((c.name, kv, st, d, fl))

    print(f"\n{'modele':20s} {'KV resident':>13s} {'KV lu/token':>13s} {'ratio':>7s}"
          f" {'FLOPs/tok':>11s} {'t_HBM(KV)':>11s} {'t_calcul':>10s}")
    print("-" * 96)
    for nm, kv, st, d, fl in rows:
        t_bw = st / h['bw'] * 1e3
        t_fl = fl / h['fp8'] * 1e3
        print(f"{nm:20s} {kv/GB:10.2f} GB {st/1e6:10.1f} MB {kv/st:6.1f}x"
              f" {fl/1e9:8.1f} G {t_bw:9.3f} ms {t_fl:8.4f} ms")

    print(f"\n  -> Lecture : 'KV resident' fixe le plafond memoire, 'KV lu/token' fixe la latence.")
    print(f"     Le ratio mesure le decouplage permis par la sparsite.")

    # ---- intensite arithmetique de la partie attention
    print(f"\n{'modele':20s} {'intensite arith. attention (FLOP/octet)':>45s}")
    print("-" * 96)
    for nm, kv, st, d, fl in rows:
        # part attention des FLOPs
        c = {"DeepSeek-V3.2": v32, "DeepSeek-V4-Flash": f, "DeepSeek-V4-Pro": p}[nm]
        act = {"DeepSeek-V3.2": 37e9, "DeepSeek-V4-Flash": 13e9, "DeepSeek-V4-Pro": 49e9}[nm]
        fn = flops_v32 if nm == "DeepSeek-V3.2" else flops_v4
        _, dd = fn(c, n, act)
        f_attn = dd['core'] + dd['idx']
        print(f"{nm:20s} {f_attn/st:45.2f}")
    ridge = h['fp8'] / h['bw']
    print(f"\n  point de bascule (ridge point) du {hw} : {ridge:.1f} FLOP/octet")
    print(f"  -> toutes ces valeurs sont TRES en dessous : l'attention en decodage est")
    print(f"     massivement limitee par la bande passante, jamais par le calcul.")


def batch_scaling(n=1_000_000, hw="B200"):
    h = HW[hw]
    print()
    print("=" * 96)
    print(f"CAPACITE ET DEBIT  --  contexte {n:,} --  {hw}")
    print("=" * 96)
    print(f"{'modele':20s} {'poids':>10s} {'GPU min':>8s} {'KV/seq':>10s}"
          f" {'seq/noeud':>10s} {'ms/token':>10s} {'tok/s noeud':>12s}")
    print("-" * 96)
    for c, kvfn, stfn, act, nr, na in (
            (v32, kv_bytes_per_token_v32, streamed_v32, 37e9, 256, 8),
            (f, kv_bytes_per_token_v4, streamed_v4, 13e9, 256, 6),
            (p, kv_bytes_per_token_v4, streamed_v4, 49e9, 384, 6)):
        W = weight_bytes(c.name)
        ngpu = int(np.ceil(W / (h['mem'] * GB * 0.85)))     # 85% utile
        ngpu = max(ngpu, 1)
        # arrondi a la puissance de 2 >= (noeud realiste)
        ngpu_node = int(2 ** np.ceil(np.log2(ngpu)))
        total_mem = ngpu_node * h['mem'] * GB
        free = total_mem - W
        kv_seq = kvfn(c) * n
        nseq = max(int(free / kv_seq), 0)

        st, _ = stfn(c, n)
        fl = (flops_v32 if c.name == "DeepSeek-V3.2" else flops_v4)(c, n, act)[0]
        # a batch B : temps = max(lecture poids + B*KV lu, calcul) / BW agregee
        BWtot = ngpu_node * h['bw']
        FLtot = ngpu_node * h['fp8']
        B = min(nseq, 64)
        if B < 1:
            print(f"{c.name:20s} {W/GB:7.0f} GB {ngpu_node:8d} {kv_seq/GB:7.2f} GB"
                  f" {nseq:10d} {'--':>10s} {'--':>12s}")
            continue
        experts = expected_experts_read(nr, na, B)
        w_read = W * (experts / nr) * 0.97 + W * 0.03      # ~97% des poids = experts
        t = max((w_read + B * st) / BWtot, B * fl / FLtot) * 1e3
        print(f"{c.name:20s} {W/GB:7.0f} GB {ngpu_node:8d} {kv_seq/GB:7.2f} GB"
              f" {nseq:10d} {t:9.2f} ms {B/t*1e3:11.0f}")
    print(f"\n  (batch plafonne a 64 sequences ; 'seq/noeud' = limite memoire pure)")


if __name__ == "__main__":
    for hw in ("H800", "B200"):
        analyse(1_000_000, hw)
        print()
    for hw in ("H800", "B200"):
        batch_scaling(1_000_000, hw)
    print()
    print("=" * 96)
    print("EVOLUTION DU 'KV LU PAR TOKEN' AVEC LE CONTEXTE (MB)")
    print("=" * 96)
    print(f"{'contexte':>10s} {'V3.2':>10s} {'V4-Flash':>10s} {'V4-Pro':>10s} {'gain Flash':>12s}")
    print("-" * 96)
    for n in (32_000, 128_000, 256_000, 512_000, 1_000_000):
        a, _ = streamed_v32(v32, n)
        b, _ = streamed_v4(f, n)
        c_, _ = streamed_v4(p, n)
        print(f"{n:10,d} {a/1e6:9.1f} {b/1e6:9.1f} {c_/1e6:9.1f} {a/b:11.1f}x")
