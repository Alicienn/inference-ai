"""
EXPERIENCE A -- Selection par blocs (CSA/HCA) vs selection par token (DSA)

Question testee : DeepSeek-V4 remplace la selection au niveau token (DSA de V3.2)
par une selection au niveau de BLOCS compresses (CSA : m=4 ; HCA : m'=128).
C'est le pari architectural central de V4. Quand est-il gagnant, quand perd-il ?

Methode : on mesure la structure REELLE de l'attention (GPT-2, cartes d'attention
completes sur du texte wikitext), puis on compare, a budget egal, la masse
d'attention recuperee par :
   (a) selection oracle des k meilleurs TOKENS          <- borne sup. de DSA
   (b) selection oracle des k/m meilleurs BLOCS de m    <- borne sup. de CSA/HCA
On compare a deux budgets differents :
   - budget "tokens touches" egal  -> cout de la granularite bloc, seule
   - budget "bande passante indexeur" egal -> ce que la compression fait GAGNER

Limite assumee : GPT-2 (117M, contexte 1024) n'est pas un modele 1M-contexte.
Les valeurs absolues ne se transposent pas ; la STRUCTURE (regroupement spatial
de la masse d'attention) est le phenomene universel qu'on cherche a quantifier.
"""
import numpy as np, torch, json, pathlib
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\07_analyses")
OUT.mkdir(exist_ok=True)

SEQ = 1024
NDOC = 6
BLOCKS = [1, 2, 4, 8, 16, 32, 64, 128]


def get_texts(n):
    """Textes longs depuis wikitext en cache ; repli sur un texte synthetique."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        buf, out = "", []
        for r in ds:
            buf += r["text"]
            if len(buf) > 12000:
                out.append(buf); buf = ""
                if len(out) >= n: return out
        return out or [buf]
    except Exception as e:
        print(f"  [wikitext indisponible: {e}] -> repli")
        return None


def collect_attention():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    texts = get_texts(NDOC)
    if texts is None:
        raise SystemExit("pas de corpus texte")
    print(f"  {len(texts)} documents charges")
    mats = []
    for i, t in enumerate(texts):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids
        if ids.shape[1] < SEQ:
            continue
        out = model(ids, output_attentions=True)
        # attentions : tuple de (1, heads, seq, seq)
        a = torch.stack([x[0] for x in out.attentions])   # (L, H, S, S)
        mats.append(a.float().numpy())
        print(f"  doc {i}: attention {a.shape}")
        if len(mats) >= 4:
            break
    return mats


def analyse(mats):
    """
    Pour chaque distribution d'attention d'une query (ligne de la matrice, causale),
    on calcule la masse recuperee par selection top-k tokens et top-(k/m) blocs.
    """
    # positions de query analysees : suffisamment loin pour avoir du contexte
    qpos = [255, 511, 767, 1023]
    res = {m: {} for m in BLOCKS}
    budgets = [16, 32, 64, 128, 256]

    counts = {m: {b: [] for b in budgets} for m in BLOCKS}
    conc = []          # concentration : masse dans le top-64
    for a in mats:
        L, H, S, _ = a.shape
        for l in range(L):
            for h in range(H):
                for q in qpos:
                    w = a[l, h, q, :q + 1]
                    if w.size < 300:
                        continue
                    s = w.sum()
                    if s <= 0:
                        continue
                    w = w / s
                    srt = np.sort(w)[::-1]
                    conc.append(srt[:64].sum())
                    for m in BLOCKS:
                        nb = w.size // m
                        if nb < 4:
                            continue
                        blk = w[:nb * m].reshape(nb, m).sum(1)
                        blk_srt = np.sort(blk)[::-1]
                        cblk = np.cumsum(blk_srt)
                        ctok = np.cumsum(srt)
                        for b in budgets:
                            # budget b = nombre de TOKENS qu'on a le droit de lire
                            nblk = max(b // m, 1)
                            if nblk > nb:
                                continue
                            mass_blk = cblk[nblk - 1]
                            mass_tok = ctok[min(b, w.size) - 1]
                            counts[m][b].append((mass_blk, mass_tok))
    return counts, np.array(conc), budgets


def main():
    print("=" * 92)
    print("EXPERIENCE A : selection par blocs vs par token, sur attention REELLE (GPT-2)")
    print("=" * 92)
    mats = collect_attention()
    counts, conc, budgets = analyse(mats)

    print(f"\nConcentration de l'attention : masse moyenne dans le top-64 tokens "
          f"= {conc.mean():.3f}  (mediane {np.median(conc):.3f})")
    print(f"  -> l'attention est effectivement tres creuse : justifie le principe meme du top-k.\n")

    print("=" * 92)
    print("1. COUT DE LA GRANULARITE BLOC  (a nombre de tokens lus EGAL)")
    print("=" * 92)
    print("   masse d'attention recuperee ; m=1 est la reference (selection par token)")
    print(f"\n{'budget':>8s} | " + " ".join(f"{'m='+str(m):>9s}" for m in BLOCKS))
    print("-" * 92)
    table = {}
    for b in budgets:
        row = []
        for m in BLOCKS:
            v = counts[m][b]
            row.append(np.mean([x[0] for x in v]) if v else np.nan)
        table[b] = row
        print(f"{b:8d} | " + " ".join(f"{x:9.4f}" for x in row))

    print(f"\n{'budget':>8s} | " + " ".join(f"{'m='+str(m):>9s}" for m in BLOCKS)
          + "   (en % de la selection par token)")
    print("-" * 92)
    for b in budgets:
        ref = table[b][0]
        print(f"{b:8d} | " + " ".join(f"{100*x/ref:8.1f}%" for x in table[b]))

    print()
    print("=" * 92)
    print("2. A BANDE PASSANTE D'INDEXEUR EGALE  (le vrai arbitrage de V4)")
    print("=" * 92)
    print("""   L'indexeur DSA (V3.2) lit n cles de dim 128 en FP8      -> 128 n octets
   L'indexeur CSA (V4)   lit n/m cles de dim 128 en FP4     -> 16 n/m octets  (m=4 -> 32x moins)
   A budget d'octets egal, CSA peut donc couvrir un contexte bien plus long,
   ou depenser le budget economise ailleurs. On compare ici la masse recuperee
   quand on accorde a la selection par bloc un budget de tokens m fois plus grand :""")
    print(f"\n{'budget tok':>11s} | {'DSA (m=1)':>10s} | " +
          " ".join(f"{'CSA m='+str(m):>11s}" for m in [2, 4, 8, 16]))
    print("-" * 92)
    for b in budgets:
        ref = table[b][0]
        cells = []
        for m in [2, 4, 8, 16]:
            bb = b * m
            v = counts[m].get(bb)
            cells.append(np.mean([x[0] for x in v]) if v else np.nan)
        print(f"{b:11d} | {ref:10.4f} | " + " ".join(f"{x:11.4f}" for x in cells))

    # sauvegarde
    js = {"concentration_top64_mean": float(conc.mean()),
          "budgets": budgets, "blocks": BLOCKS,
          "mass_by_budget_and_block": {str(b): [float(x) for x in table[b]] for b in budgets}}
    (OUT / "expA_block_vs_token.json").write_text(json.dumps(js, indent=2))
    print(f"\n  resultats -> {OUT/'expA_block_vs_token.json'}")


if __name__ == "__main__":
    main()
