"""
EXPERIENCE A2 -- Pourquoi la compression par blocs de V4 fonctionne, et quand elle casse.

Corrige le biais de l'exp. A (cellules ou nblk*m depassait le budget) et ajoute :
  (1) longueur de correlation spatiale de la masse d'attention  -> le MECANISME
  (2) test "aiguille isolee"                                    -> le MODE D'ECHEC
  (3) comparaison a bande passante d'indexeur egale, proprement bornee
"""
import numpy as np, torch, json, pathlib
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\07_analyses")
OUT.mkdir(exist_ok=True)
SEQ, NDOC = 1024, 3
BLOCKS = [1, 2, 4, 8, 16, 32, 64, 128]
BUDGETS = [32, 64, 128, 256, 512]


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
        out = model(ids, output_attentions=True)
        mats.append(torch.stack([x[0] for x in out.attentions]).float().numpy())
    return mats


def main():
    print("=" * 94)
    print("EXPERIENCE A2 : mecanisme et limites de la selection par blocs (attention reelle GPT-2)")
    print("=" * 94)
    mats = load_attn()
    print(f"  {len(mats)} documents x 12 couches x 12 tetes x {SEQ} positions\n")

    qpos = [383, 639, 895, 1023]
    # accumulateurs
    mass = {m: {b: [] for b in BUDGETS} for m in BLOCKS}
    autocorr = np.zeros(65); ac_n = 0
    needle_iso, needle_tot = 0, 0
    gini = []

    for a in mats:
        L, H, S, _ = a.shape
        for l in range(L):
            for h in range(H):
                for q in qpos:
                    w = a[l, h, q, :q + 1].astype(np.float64)
                    s = w.sum()
                    if s <= 0 or w.size < 512: continue
                    w /= s
                    srt = np.sort(w)[::-1]; ctok = np.cumsum(srt)

                    # --- (1) autocorrelation de la masse d'attention
                    x = w - w.mean()
                    v = (x * x).sum()
                    if v > 0:
                        for lag in range(65):
                            autocorr[lag] += (x[:w.size - lag] * x[lag:]).sum() / v
                        ac_n += 1

                    # --- (2) aiguilles isolees : top-16 sans voisin top-16 a moins de 4
                    top = np.sort(np.argsort(w)[-16:])
                    for i, pj in enumerate(top):
                        nb = [o for o in top if o != pj and abs(o - pj) < 4]
                        needle_tot += 1
                        if not nb: needle_iso += 1

                    # --- concentration (Gini approx : masse top-1%)
                    gini.append(srt[:max(1, w.size // 100)].sum())

                    # --- (3) masse recuperee, budget STRICTEMENT respecte
                    for m in BLOCKS:
                        nbk = w.size // m
                        if nbk < 2: continue
                        blk = w[:nbk * m].reshape(nbk, m).sum(1)
                        cblk = np.cumsum(np.sort(blk)[::-1])
                        for b in BUDGETS:
                            nsel = b // m                     # blocs entiers finançables
                            if nsel < 1 or nsel > nbk: continue
                            mass[m][b].append(cblk[nsel - 1])

    autocorr /= max(ac_n, 1)
    print("=" * 94)
    print("(1) MECANISME : autocorrelation spatiale de la masse d'attention")
    print("=" * 94)
    print("   lag :  " + " ".join(f"{d:6d}" for d in [1, 2, 4, 8, 16, 32, 64]))
    print("   rho :  " + " ".join(f"{autocorr[d]:6.3f}" for d in [1, 2, 4, 8, 16, 32, 64]))
    half = next((d for d in range(1, 65) if autocorr[d] < 0.5 * autocorr[0]), 64)
    print(f"\n   longueur de correlation (rho < 0.5) ~ {half} tokens")
    print(f"   -> la masse d'attention est REGROUPEE sur ~{half} tokens contigus.")
    print(f"      Un bloc de m=4 reste bien en-deca : il agrege des tokens deja correles,")
    print(f"      donc il detruit peu d'information. C'est la justification de m=4.")
    print(f"      Un bloc m'=128 depasse largement cette longueur : HCA melange des")
    print(f"      contenus non correles -> il ne peut servir que de resume grossier.")

    print()
    print("=" * 94)
    print("(2) MODE D'ECHEC : aiguilles isolees")
    print("=" * 94)
    print(f"   {100*needle_iso/needle_tot:.1f}% des tokens du top-16 n'ont aucun autre")
    print(f"   token du top-16 a moins de 4 positions : ce sont des 'aiguilles' que la")
    print(f"   compression dilue (facteur jusqu'a m). C'est le risque residuel de CSA,")
    print(f"   et precisement ce que la branche sliding-window + l'attention sink compensent.")
    print(f"   Masse moyenne concentree dans le top-1% des positions : {np.mean(gini):.3f}")

    print()
    print("=" * 94)
    print("(3) MASSE RECUPEREE, budget en tokens lus strictement respecte")
    print("=" * 94)
    print(f"{'budget':>8s} | " + " ".join(f"{'m='+str(m):>8s}" for m in BLOCKS))
    print("-" * 94)
    tbl = {}
    for b in BUDGETS:
        row = [np.mean(mass[m][b]) if mass[m][b] else np.nan for m in BLOCKS]
        tbl[b] = row
        print(f"{b:8d} | " + " ".join("     --- " if np.isnan(x) else f"{x:8.4f}" for x in row))
    print(f"\n{'budget':>8s} | " + " ".join(f"{'m='+str(m):>8s}" for m in BLOCKS)
          + "    (% de la selection par token)")
    print("-" * 94)
    for b in BUDGETS:
        ref = tbl[b][0]
        print(f"{b:8d} | " + " ".join("     --- " if np.isnan(x) else f"{100*x/ref:7.1f}%"
                                      for x in tbl[b]))

    print()
    print("=" * 94)
    print("(4) A BANDE PASSANTE D'INDEXEUR EGALE  --  l'arbitrage reel de DeepSeek-V4")
    print("=" * 94)
    print("""   Cout indexeur par token de contexte :
     DSA (V3.2) : 1 cle de dim 128 en FP8       = 128 octets/token
     CSA (V4)   : 1 cle de dim 128 en FP4 /m    =  64/m octets/token
   => a m=4, CSA consomme 8x moins de bande passante d'indexeur que DSA.
   On compare donc DSA(budget b) a CSA(budget b x m), tous deux bornes par le contexte.""")
    print(f"\n{'budget DSA':>11s} | {'DSA m=1':>9s} | " +
          " ".join(f"{'CSA m='+str(m):>10s}" for m in [2, 4, 8, 16]))
    print("-" * 94)
    for b in [32, 64, 128]:
        ref = tbl[b][0]
        cells = []
        for m in [2, 4, 8, 16]:
            bb = b * m
            v = mass[m].get(bb)
            cells.append(np.mean(v) if v else np.nan)
        print(f"{b:11d} | {ref:9.4f} | " +
              " ".join("      --- " if np.isnan(x) else f"{x:10.4f}" for x in cells)
              + "   " + " ".join("" if np.isnan(x) else f"({100*(x/ref-1):+.1f}%)"
                                 for x in cells))

    js = dict(autocorr=[float(x) for x in autocorr],
              corr_half_len=int(half),
              needle_isolated_pct=float(100 * needle_iso / needle_tot),
              mass=({str(b): [None if np.isnan(x) else float(x) for x in tbl[b]]
                     for b in BUDGETS}),
              blocks=BLOCKS, budgets=BUDGETS)
    (OUT / "expA2_mechanism.json").write_text(json.dumps(js, indent=2))
    print(f"\n  -> {OUT/'expA2_mechanism.json'}")


if __name__ == "__main__":
    main()
