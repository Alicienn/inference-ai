"""
EXPERIENCE E -- Validation croisee sur une seconde architecture.

La limite la plus serieuse des experiences A2/A3/C est qu'elles reposent sur GPT-2
(117M, contexte 1024, positions apprises, attention dense classique). On replique ici
les trois resultats structurels centraux sur SmolLM2-135M, architecture MODERNE
(RoPE, GQA 9/3, 30 couches, SwiGLU) -- famille differente, entrainement different.

Si les trois resultats tiennent sur les deux modeles, ils sont structurels et non des
artefacts de GPT-2. S'ils divergent, il faut le dire.
"""
import numpy as np, torch, json, pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
OUT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\07_analyses")
MID = "HuggingFaceTB/SmolLM2-135M"
SEQ, NDOC = 1024, 3
BLOCKS = [1, 2, 4, 8, 16, 32]
BUDGETS = [32, 64, 128, 256]
GAMMAS = [1, 2, 4, 8]
TOPK = 64


def load_attn():
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(
        MID, dtype=torch.float32, attn_implementation="eager").eval()
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
    print("EXPERIENCE E : validation croisee sur SmolLM2-135M (RoPE, GQA) vs GPT-2")
    print("=" * 94)
    mats = load_attn()
    L, H, S, _ = mats[0].shape
    print(f"  {len(mats)} docs x {L} couches x {H} tetes x {S} positions\n")

    qpos = [383, 639, 895, 1023]
    autocorr = np.zeros(65); ac_n = 0
    needle_iso = needle_tot = 0
    conc64, conc1pct = [], []
    mass = {m: {b: [] for b in BUDGETS} for m in BLOCKS}
    union = {m: {g: [] for g in GAMMAS} for m in (1, 4)}

    for a in mats:
        for l in range(L):
            for h in range(H):
                for q in qpos:
                    w = a[l, h, q, :q + 1].astype(np.float64)
                    s = w.sum()
                    if s <= 0 or w.size < 512: continue
                    w /= s
                    srt = np.sort(w)[::-1]
                    conc64.append(srt[:64].sum())
                    conc1pct.append(srt[:max(1, w.size // 100)].sum())
                    x = w - w.mean(); v = (x * x).sum()
                    if v > 0:
                        for lag in range(65):
                            autocorr[lag] += (x[:w.size - lag] * x[lag:]).sum() / v
                        ac_n += 1
                    top = np.sort(np.argsort(w)[-16:])
                    for pj in top:
                        needle_tot += 1
                        if not [o for o in top if o != pj and abs(o - pj) < 4]:
                            needle_iso += 1
                    for m in BLOCKS:
                        nbk = w.size // m
                        if nbk < 2: continue
                        blk = w[:nbk * m].reshape(nbk, m).sum(1)
                        cblk = np.cumsum(np.sort(blk)[::-1])
                        for b in BUDGETS:
                            ns = b // m
                            if 1 <= ns <= nbk:
                                mass[m][b].append(cblk[ns - 1])
        # recouvrement des selections (spec x sparse)
        for l in range(0, L, 4):
            for h in range(0, H, 3):
                for q0 in (500, 700, 900):
                    for m in (1, 4):
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
                            union[m][g].append(len(set().union(*sets[:g + 1])) / TOPK)

    autocorr /= max(ac_n, 1)
    print("=" * 94)
    print("RESULTAT 1 : concentration et absence de regroupement spatial")
    print("=" * 94)
    print(f"{'':34s} {'SmolLM2-135M':>14s} {'GPT-2 (rappel)':>16s}")
    print("-" * 94)
    print(f"{'masse dans le top-64':34s} {np.mean(conc64):14.3f} {0.876:16.3f}")
    print(f"{'masse dans le top-1% des positions':34s} {np.mean(conc1pct):14.3f} {0.655:16.3f}")
    print(f"{'autocorrelation au decalage 1':34s} {autocorr[1]:14.3f} {0.207:16.3f}")
    print(f"{'autocorrelation au decalage 4':34s} {autocorr[4]:14.3f} {0.130:16.3f}")
    print(f"{'aiguilles isolees (top-16)':34s} {100*needle_iso/needle_tot:13.1f}% {30.4:15.1f}%")

    print()
    print("=" * 94)
    print("RESULTAT 2 : cout de la granularite bloc (% de la selection par token)")
    print("=" * 94)
    print(f"{'budget':>8s} | " + " ".join(f"{'m='+str(m):>8s}" for m in BLOCKS))
    print("-" * 94)
    tbl = {}
    for b in BUDGETS:
        row = [np.mean(mass[m][b]) if mass[m][b] else np.nan for m in BLOCKS]
        tbl[b] = row
        ref = row[0]
        print(f"{b:8d} | " + " ".join("     --- " if np.isnan(x) else f"{100*x/ref:7.1f}%"
                                      for x in row))
    print(f"\n  (GPT-2, rappel a m=4 : 93,7% / 95,3% / 96,6% / 97,9% pour budgets 32/64/128/256)")

    print()
    print("=" * 94)
    print("RESULTAT 3 : a bande passante d'indexeur egale (DSA b vs CSA b*m)")
    print("=" * 94)
    print(f"{'budget':>8s} | {'DSA m=1':>9s} | {'CSA m=2':>9s} {'CSA m=4':>9s} {'CSA m=8':>9s}")
    print("-" * 94)
    for b in [32, 64]:
        ref = tbl[b][0]
        cells = []
        for m in (2, 4, 8):
            v = mass[m].get(b * m)
            cells.append(np.mean(v) if v else np.nan)
        print(f"{b:8d} | {ref:9.4f} | " +
              " ".join("     --- " if np.isnan(x) else f"{x:9.4f}" for x in cells) +
              "  " + " ".join("" if np.isnan(x) else f"({100*(x/ref-1):+.1f}%)" for x in cells))

    print()
    print("=" * 94)
    print("RESULTAT 4 : recouvrement des selections (decodage speculatif x sparsite)")
    print("=" * 94)
    print(f"{'gamma':>6s} | {'m=1 amortis.':>13s} {'m=4 amortis.':>13s} | {'GPT-2 m=1':>10s} {'GPT-2 m=4':>10s}")
    print("-" * 94)
    ref_gpt2 = {1: (76.8, 87.3), 2: (65.3, 80.5), 4: (53.5, 72.8), 8: (41.8, 65.2)}
    for g in GAMMAS:
        u1 = np.mean(union[1][g]) if union[1][g] else np.nan
        u4 = np.mean(union[4][g]) if union[4][g] else np.nan
        a1, a4 = 100 / u1, 100 / u4
        r = ref_gpt2[g]
        print(f"{g:6d} | {a1:12.1f}% {a4:12.1f}% | {r[0]:9.1f}% {r[1]:9.1f}%")

    js = dict(model=MID, conc_top64=float(np.mean(conc64)),
              conc_top1pct=float(np.mean(conc1pct)),
              autocorr=[float(x) for x in autocorr[:17]],
              needle_isolated_pct=float(100 * needle_iso / needle_tot),
              mass={str(b): [None if np.isnan(x) else float(x) for x in tbl[b]] for b in BUDGETS},
              blocks=BLOCKS, budgets=BUDGETS,
              union={str(m): {str(g): float(np.mean(union[m][g])) if union[m][g] else None
                              for g in GAMMAS} for m in (1, 4)})
    (OUT / "expE_crossvalidation.json").write_text(json.dumps(js, indent=2))
    print(f"\n  -> {OUT/'expE_crossvalidation.json'}")


if __name__ == "__main__":
    main()
