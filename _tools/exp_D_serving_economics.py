"""
EXPERIENCE D -- Economie du serving : la compression du KV est-elle un PREREQUIS
a la desagregation prefill/decode en contexte long ?

These testee : DistServe/Mooncake separent prefill et decode sur des machines
dediees, ce qui impose de TRANSFERER le KV cache du noeud de prefill vers le noeud
de decode. Ce transfert croit avec la taille du cache. On verifie si, aux tailles
de cache de V3.2, ce transfert devient prohibitif -- auquel cas la compression
architecturale du KV n'est pas un simple gain de memoire mais la condition de
possibilite d'une technique de serving.
"""
import numpy as np
from model_inference_cost import (v32, f, p, kv_bytes_per_token_v32,
                                  kv_bytes_per_token_v4, flops_v32, flops_v4, GB)

LINKS = {
    "NVLink 4 (intra-noeud)": 900e9,
    "InfiniBand NDR 400G": 50e9,
    "Ethernet 200G": 25e9,
    "Ethernet 100G": 12.5e9,
}
HW_FP8 = 1979e12       # H800
MFU = 0.40


def prefill_time(cfg, n, act, ngpu):
    """Temps de prefill approx : FLOPs cumules sur les n positions."""
    fn = flops_v32 if cfg.name == "DeepSeek-V3.2" else flops_v4
    # integrale approchee du cout par position (l'attention croit avec la position)
    tot = 0.0
    for frac in np.linspace(0.02, 1.0, 25):
        tot += fn(cfg, max(int(n * frac), 1), act)[0] * (n / 25)
    return tot / (ngpu * HW_FP8 * MFU)


def main():
    n = 1_000_000
    print("=" * 96)
    print(f"EXPERIENCE D : cout du transfert KV en desagregation prefill/decode (contexte {n:,})")
    print("=" * 96)

    models = [("DeepSeek-V3.2", v32, kv_bytes_per_token_v32(v32) * n, 37e9, 16),
              ("DeepSeek-V4-Flash", f, kv_bytes_per_token_v4(f) * n, 13e9, 4),
              ("DeepSeek-V4-Pro", p, kv_bytes_per_token_v4(p) * n, 49e9, 16)]

    print(f"\n{'modele':20s} {'KV a transferer':>16s} | " +
          " ".join(f"{k.split(' (')[0][:14]:>14s}" for k in LINKS))
    print("-" * 96)
    for nm, cfg, kv, act, ng in models:
        cells = [kv / bw * 1e3 for bw in LINKS.values()]
        print(f"{nm:20s} {kv/GB:13.2f} GB | " + " ".join(f"{c:12.0f}ms" for c in cells))

    print(f"\n{'modele':20s} {'prefill (s)':>12s} {'transfert IB':>14s} {'part du transfert':>19s}")
    print("-" * 96)
    for nm, cfg, kv, act, ng in models:
        tp = prefill_time(cfg, n, act, ng)
        tt = kv / LINKS["InfiniBand NDR 400G"]
        print(f"{nm:20s} {tp:11.2f}s {tt*1e3:12.0f}ms {100*tt/(tp+tt):17.1f}%")

    print()
    print("=" * 96)
    print("LECTURE")
    print("=" * 96)
    kv32 = kv_bytes_per_token_v32(v32) * n
    kvf = kv_bytes_per_token_v4(f) * n
    print(f"""  A 1M de contexte, V3.2 doit deplacer {kv32/GB:.0f} GB entre le noeud de prefill et
  le noeud de decode, soit {kv32/LINKS['InfiniBand NDR 400G']:.1f} s sur InfiniBand 400G. V4-Flash n'en
  transfere que {kvf/GB:.1f} GB ({kv32/kvf:.0f}x moins), soit {kvf/LINKS['InfiniBand NDR 400G']*1e3:.0f} ms.

  ATTENTION -- l'hypothese intuitive selon laquelle ce transfert interdirait la
  desagregation est FAUSSE a prefill froid : le prefill de 1M coute ici ~{prefill_time(v32, n, 37e9, 16):.0f} s,
  et {kv32/LINKS['InfiniBand NDR 400G']:.1f} s de transfert n'en representent que {100*(kv32/LINKS['InfiniBand NDR 400G'])/(prefill_time(v32, n, 37e9, 16)+kv32/LINKS['InfiniBand NDR 400G']):.1f}%. Le calcul domine largement.
  Le transfert ne devient decisif que sous REUTILISATION DE PREFIXE (section suivante),
  ou il n'y a plus de prefill a amortir.""")

    # ---- REGIME DECISIF : reutilisation de prefixe (cache hit)
    print()
    print("=" * 96)
    print("LE REGIME OU LE TRANSFERT DEVIENT DECISIF : REUTILISATION DE PREFIXE")
    print("=" * 96)
    print("""  Le calcul ci-dessus suppose un prefill A FROID. Or Mooncake (2407.00079) est
  explicitement 'KVCache-centric' : son gain vient de la REUTILISATION de prefixes
  deja calcules (conversations multi-tours, agents, RAG). En cas de hit, il n'y a
  PLUS de prefill du tout -- il ne reste que le transfert. Le transfert n'est alors
  pas 2% du cout : il est 100% du cout, et devient le facteur limitant.""")
    print(f"\n  Temps pour servir un prefixe de 1M deja en cache, puis 1er token :")
    print(f"\n{'modele':20s} {'via NVLink':>12s} {'via IB 400G':>13s} {'via Eth 100G':>14s}"
          f" {'TTFT acceptable ?':>20s}")
    print("-" * 96)
    for nm, cfg, kv, act, ng in models:
        t = [kv / LINKS[k] * 1e3 for k in
             ("NVLink 4 (intra-noeud)", "InfiniBand NDR 400G", "Ethernet 100G")]
        ok = "oui" if t[1] < 500 else "non (IB > 500ms)"
        print(f"{nm:20s} {t[0]:10.0f}ms {t[1]:11.0f}ms {t[2]:12.0f}ms {ok:>20s}")
    print("""
  => C'est ici que la compression du KV change la nature du systeme : avec V3.2,
     un hit de cache sur 1M coute ~0.9 s de transfert inter-noeud, ce qui annule
     presque l'interet du cache. Avec V4-Flash (66 ms), la reutilisation de
     prefixe redevient franchement rentable.""")

    # ---- dependance a la longueur de contexte
    print()
    print("=" * 96)
    print("PART DU TRANSFERT DANS LE PREFILL A FROID, SELON LE CONTEXTE (IB 400G)")
    print("=" * 96)
    print(f"{'contexte':>10s} | {'V3.2':>10s} {'V4-Flash':>10s}")
    print("-" * 96)
    for nn in (8_000, 32_000, 128_000, 512_000, 1_000_000):
        row = []
        for nm, cfg, _, act, ng in models[:2]:
            kvf_ = (kv_bytes_per_token_v32(v32) if nm == "DeepSeek-V3.2"
                    else kv_bytes_per_token_v4(f)) * nn
            tp = prefill_time(cfg, nn, act, ng)
            tt = kvf_ / LINKS["InfiniBand NDR 400G"]
            row.append(100 * tt / (tp + tt))
        print(f"{nn:10,d} | {row[0]:9.1f}% {row[1]:9.1f}%")
    print("""
  => A froid, le transfert reste minoritaire a toutes les longueurs : le prefill
     domine. L'hypothese initiale ('le transfert KV interdit la desagregation')
     est donc FAUSSE dans ce regime, et n'est vraie que sous reutilisation de cache.""")

    # -- seuil : jusqu'ou peut-on desagreger ?
    print()
    print("=" * 96)
    print("SEUIL : taille de KV cache max pour que le transfert reste < 10% du prefill")
    print("=" * 96)
    print(f"{'lien':26s} {'contexte':>10s} {'KV max admissible':>20s}")
    print("-" * 96)
    for lname, bw in LINKS.items():
        tp = prefill_time(f, n, 13e9, 4)
        kvmax = 0.10 / 0.90 * tp * bw
        print(f"{lname:26s} {n:10,d} {kvmax/GB:17.1f} GB")
    print(f"\n  (calcule sur le prefill de V4-Flash sur 4 GPU H800 a {MFU:.0%} de MFU)")


if __name__ == "__main__":
    main()
