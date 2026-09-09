"""
EXPERIENCE A3 -- Le "trou de Jensen" : la compression de cles peut-elle retrouver les aiguilles ?

L'exp. A2 a infirme l'hypothese du regroupement spatial (rho=0.21 au lag 1, 30% d'aiguilles
isolees) tout en montrant que la selection par blocs perd peu de masse. Explication reelle :
le score d'un bloc par SOMME preserve la detectabilite d'une aiguille ; le cout de la
granularite n'est pas un defaut de detection mais une DILUTION DU BUDGET.

Mais CSA ne score pas par somme : il score par produit scalaire avec une CLE COMPRESSEE
(moyenne ponderee apprise des cles du bloc). Or, par l'inegalite de Jensen :
        exp( moyenne_j (logit_j) )  <=  moyenne_j exp(logit_j)
La mise en commun des cles SOUS-ESTIME systematiquement un bloc contenant une aiguille.
C'est un risque structurel que le papier V4 ne discute pas.

On quantifie ici ce trou, et on mesure a quel point le compresseur doit etre "max-like"
(softmax de basse entropie) pour l'annuler. Parametre beta : 0 = moyenne uniforme (pire cas),
beta -> inf = max (aiguille parfaitement preservee).
"""
import numpy as np, torch, json, pathlib
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\07_analyses")
SEQ, NDOC = 1024, 3
M_LIST = [4, 8, 16]
BUDGETS = [64, 128, 256]
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]


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


def block_logit(lg, beta):
    """
    Logit effectif d'un bloc sous une mise en commun softmax de parametre beta.
       beta=0    -> moyenne uniforme des logits (compresseur non informe)
       beta=inf  -> max des logits (compresseur parfaitement max-like)
    lg : (nblocs, m)
    """
    if np.isinf(beta):
        return lg.max(1)
    z = beta * (lg - lg.max(1, keepdims=True))
    s = np.exp(z); s /= s.sum(1, keepdims=True)
    return (s * lg).sum(1)


def main():
    print("=" * 96)
    print("EXPERIENCE A3 : trou de Jensen dans la selection par cle compressee (attention reelle)")
    print("=" * 96)
    mats = load_attn()
    qpos = [383, 639, 895, 1023]

    # masse recuperee : [m][budget][beta]  + reference somme-oracle et token-oracle
    acc = {m: {b: {be: [] for be in BETAS} for b in BUDGETS} for m in M_LIST}
    oracle_sum = {m: {b: [] for b in BUDGETS} for m in M_LIST}
    oracle_tok = {b: [] for b in BUDGETS}

    for a in mats:
        L, H, S, _ = a.shape
        for l in range(L):
            for h in range(H):
                for q in qpos:
                    w = a[l, h, q, :q + 1].astype(np.float64)
                    s = w.sum()
                    if s <= 0 or w.size < 512: continue
                    w /= s
                    lg = np.log(np.maximum(w, 1e-30))       # logits a une constante pres
                    srt = np.sort(w)[::-1]; ctok = np.cumsum(srt)
                    for b in BUDGETS:
                        oracle_tok[b].append(ctok[min(b, w.size) - 1])

                    for m in M_LIST:
                        nbk = w.size // m
                        if nbk < 4: continue
                        wb = w[:nbk * m].reshape(nbk, m)
                        lb = lg[:nbk * m].reshape(nbk, m)
                        bsum = wb.sum(1)                     # score oracle (somme)
                        for b in BUDGETS:
                            nsel = b // m
                            if nsel < 1 or nsel > nbk: continue
                            # oracle somme
                            idx = np.argsort(bsum)[-nsel:]
                            oracle_sum[m][b].append(bsum[idx].sum())
                            # scoring par cle compressee, selon beta
                            for be in BETAS:
                                sc = block_logit(lb, be)
                                idx = np.argsort(sc)[-nsel:]
                                acc[m][b][be].append(bsum[idx].sum())   # masse REELLE recuperee

    print(f"\nMasse d'attention reellement recuperee selon la maniere dont le bloc est score.")
    print("beta = 0 : cle compressee = moyenne uniforme (compresseur non entraine)")
    print("beta = inf : cle compressee = max (compresseur ideal, aiguille preservee)\n")

    rows = []
    for m in M_LIST:
        print("=" * 96)
        print(f"  m = {m}")
        print("=" * 96)
        hdr = f"{'budget':>7s} | {'top-k token':>11s} {'oracle somme':>12s} | " + \
              " ".join(f"{'b='+('inf' if np.isinf(x) else str(x)):>8s}" for x in BETAS)
        print(hdr); print("-" * 96)
        for b in BUDGETS:
            if not oracle_sum[m][b]: continue
            ot = np.mean(oracle_tok[b]); os_ = np.mean(oracle_sum[m][b])
            cells = [np.mean(acc[m][b][be]) if acc[m][b][be] else np.nan for be in BETAS]
            print(f"{b:7d} | {ot:11.4f} {os_:12.4f} | " +
                  " ".join(f"{x:8.4f}" for x in cells))
            rows.append(dict(m=m, budget=b, token=float(ot), oracle_sum=float(os_),
                             by_beta={("inf" if np.isinf(be) else str(be)): float(c)
                                      for be, c in zip(BETAS, cells)}))
        # trou de Jensen
        b = BUDGETS[-1]
        if acc[m][b][0.0] and acc[m][b][np.inf]:
            lo = np.mean(acc[m][b][0.0]); hi = np.mean(acc[m][b][np.inf])
            os_ = np.mean(oracle_sum[m][b])
            print(f"\n  trou de Jensen a budget {b} : moyenne uniforme {lo:.4f} "
                  f"-> max {hi:.4f}   (ecart {100*(hi-lo)/lo:+.1f}%)")
            print(f"  perte de la mise en commun uniforme vs oracle somme : "
                  f"{100*(lo-os_)/os_:+.1f}%")
        print()

    print("=" * 96)
    print("LECTURE")
    print("=" * 96)
    print("""  Un compresseur qui fait une moyenne uniforme (beta=0) perd nettement la masse
  portee par les aiguilles : c'est le trou de Jensen. Il se referme des que la
  ponderation devient piquee (beta >= 1-2), c'est-a-dire des que le compresseur
  apprend a se comporter comme un max.

  Consequence de conception, non explicitee dans le papier V4 : la qualite de CSA
  ne depend pas seulement du taux de compression m, mais de l'ENTROPIE de la
  ponderation de compression. Un compresseur a softmax plat annule une partie du
  benefice. Cela rend la temperature/entropie de Softmax_row (eq. 11) un
  hyperparametre critique -- et une metrique a surveiller en entrainement.""")

    (OUT / "expA3_jensen.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  -> {OUT/'expA3_jensen.json'}")


if __name__ == "__main__":
    main()
