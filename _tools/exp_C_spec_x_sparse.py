"""
EXPERIENCE C -- Interaction entre decodage speculatif et attention creuse.

Motivation. En contexte long, la latence de decodage est dominee par la lecture du
KV cache (montre dans roofline_decode.py : 7.9 GB/token pour V3.2 @1M). Le decodage
speculatif verifie gamma+1 tokens candidats en une passe : en attention DENSE, le KV
cache n'est lu QU'UNE FOIS pour tous les candidats -> le cout par token est divise par
le nombre de tokens acceptes. C'est pourquoi la speculation est bien plus payante en
contexte long qu'on ne le dit d'habitude.

Mais en attention CREUSE, chaque position candidate选 selectionne son PROPRE top-k.
La passe de verification doit lire l'UNION des selections. Si les selections divergent,
l'amortissement s'effondre. DeepSeek-V4 combine precisement les deux (MTP depth 1 + CSA),
donc cette tension est active dans leur conception.

On mesure ici, sur de l'attention REELLE, le taux de recouvrement des selections top-k
entre positions de requete voisines, puis on en deduit l'amortissement effectif.
"""
import numpy as np, torch, json, pathlib
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\07_analyses")
SEQ, NDOC = 1024, 3
GAMMAS = [1, 2, 3, 4, 6, 8]
TOPK = 64            # budget de blocs selectionnes
M_LIST = [1, 4, 16]  # granularite : 1 = token (DSA), 4 = CSA, 16


def load_attn():
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    buf, texts = "", []
    for r in ds:
        buf += r["text"]
        if len(buf) > 12000:
            texts.append(buf); buf = ""
            if len(texts) >= NDOC: break
    mats = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=SEQ).input_ids
        if ids.shape[1] < SEQ: continue
        mats.append(torch.stack([x[0] for x in model(ids, output_attentions=True).attentions])
                    .float().numpy())
    return mats


def spec_speedup(alpha, gamma, c_draft, c_verify_ratio=1.0):
    """
    Nombre attendu de tokens acceptes par passe (Leviathan et al. 2023) :
        E = (1 - alpha^(gamma+1)) / (1 - alpha)
    Cout d'une passe = gamma * c_draft + c_verify_ratio (verification de gamma+1 tokens).
    Acceleration relative au decodage autoregressif (cout 1 par token).
    """
    E = (1 - alpha ** (gamma + 1)) / (1 - alpha) if alpha < 1 else gamma + 1
    return E / (gamma * c_draft + c_verify_ratio)


def main():
    print("=" * 96)
    print("EXPERIENCE C : decodage speculatif x attention creuse")
    print("=" * 96)
    mats = load_attn()
    qbase = [500, 700, 900]

    # union[m][gamma] = |union des top-k sur gamma+1 positions| / k
    union = {m: {g: [] for g in GAMMAS} for m in M_LIST}
    for a in mats:
        L, H, S, _ = a.shape
        for l in range(L):
            for h in range(H):
                for q0 in qbase:
                    for m in M_LIST:
                        sets = []
                        for d in range(max(GAMMAS) + 1):
                            q = q0 + d
                            if q >= S: break
                            w = a[l, h, q, :q + 1].astype(np.float64)
                            nb = w.size // m
                            if nb < TOPK + 2: break
                            blk = w[:nb * m].reshape(nb, m).sum(1)
                            sets.append(set(np.argsort(blk)[-TOPK:].tolist()))
                        if len(sets) <= max(GAMMAS): continue
                        for g in GAMMAS:
                            u = set().union(*sets[:g + 1])
                            union[m][g].append(len(u) / TOPK)

    print(f"\nTaille de l'UNION des selections top-{TOPK} sur gamma+1 positions voisines")
    print("(1.0 = selections identiques -> amortissement parfait ;")
    print(" gamma+1 = selections disjointes -> aucun amortissement)\n")
    print(f"{'gamma':>6s} | " + " ".join(f"{'m='+str(m):>18s}" for m in M_LIST))
    print("-" * 96)
    U = {}
    for g in GAMMAS:
        row = []
        for m in M_LIST:
            v = np.mean(union[m][g]) if union[m][g] else np.nan
            row.append(v)
        U[g] = row
        print(f"{g:6d} | " + " ".join(f"{v:8.2f}x (max {g+1:2d})" for v in row))

    print()
    print("=" * 96)
    print("AMORTISSEMENT EFFECTIF DE LA LECTURE DU KV CACHE")
    print("=" * 96)
    print("""  En dense, verifier gamma+1 tokens ne lit le cache qu'une fois : cout relatif 1.
  En creux, il faut lire l'union : cout relatif = |union|/k. L'efficacite
  d'amortissement est donc (gamma+1) / (|union|/k) rapporte a (gamma+1).""")
    print(f"\n{'gamma':>6s} | {'dense':>8s} | " +
          " ".join(f"{'m='+str(m):>10s}" for m in M_LIST) + "   (efficacite d'amortissement)")
    print("-" * 96)
    for g in GAMMAS:
        cells = [(g + 1) / U[g][i] / (g + 1) for i in range(len(M_LIST))]
        print(f"{g:6d} | {1.0:8.2f} | " + " ".join(f"{c:10.1%}" for c in cells))

    print()
    print("=" * 96)
    print("CONSEQUENCE SUR L'ACCELERATION REELLE (modele de cout borne par la bande passante)")
    print("=" * 96)
    print("""  On suppose le decodage limite par la lecture du KV (vrai en contexte long, cf.
  roofline_decode.py). Cout de verification relatif = |union|/k au lieu de 1.
  alpha = taux d'acceptation ; c_draft = cout du brouillon (0.1 pour une tete type EAGLE-3).""")
    for alpha in (0.7, 0.8, 0.9):
        print(f"\n  alpha = {alpha}")
        print(f"  {'gamma':>6s} | {'dense':>8s} | " +
              " ".join(f"{'creux m='+str(m):>12s}" for m in M_LIST))
        print("  " + "-" * 92)
        for g in GAMMAS:
            d = spec_speedup(alpha, g, 0.1, 1.0)
            cs = [spec_speedup(alpha, g, 0.1, U[g][i]) for i in range(len(M_LIST))]
            print(f"  {g:6d} | {d:8.2f}x | " + " ".join(f"{c:11.2f}x" for c in cs))

    js = dict(topk=TOPK, gammas=GAMMAS, m_list=M_LIST,
              union_ratio={str(g): [float(x) for x in U[g]] for g in GAMMAS})
    (OUT / "expC_spec_sparse.json").write_text(json.dumps(js, indent=2))
    print(f"\n  -> {OUT/'expC_spec_sparse.json'}")


if __name__ == "__main__":
    main()
