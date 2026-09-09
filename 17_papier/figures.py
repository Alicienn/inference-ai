"""
Génération des figures du papier ASP.
Les données proviennent des mesures des sessions 4 à 10 ; celles disponibles en JSON
sont relues, les autres (mesures distantes sur T4 / A100 / H200) sont consignées ici
telles qu'elles ont été relevées, avec la session d'origine en commentaire.
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = pathlib.Path(__file__).resolve().parent / "figs"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"dense": "#444444", "asp": "#0072B2", "fused": "#D55E00",
     "igpu": "#009E73", "t4": "#0072B2", "a100": "#D55E00", "h200": "#CC79A7"}

# ---------------------------------------------------------------- données mesurées
# Session 4-7 : ratio débit gather / débit contigu, par taille de morceau
GATHER = {
    "AMD gfx1152 (LPDDR5, 80 GB/s)": ([64, 128, 256, 512, 1024, 2048, 4096, 16384],
                                      [0.67, 0.69, 0.73, 0.83, 0.92, 1.01, 1.09, 0.99]),
    "Tesla T4 (GDDR6, 274 GB/s)":    ([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                                      [1.002, 0.943, 0.908, 0.880, 0.988, 0.949, 0.988, 1.002, 0.994]),
    "A100 PCIe (HBM2e, 1372 GB/s)":  ([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                                      [1.000, 1.000, 1.000, 1.000, 0.984, 0.972, 0.980, 0.955, 1.000]),
}
# fraction du débit crête atteinte par le chemin contigu (régime de saturation)
SAT = {
    "AMD gfx1152 (LPDDR5, 80 GB/s)": [0.06, 0.12, 0.23, 0.37, 0.49, 0.75, 0.81, 1.00],
    "Tesla T4 (GDDR6, 274 GB/s)":    [0.05, 0.15, 0.30, 0.61, 0.86, 1.00, 0.99, 0.98, 0.99],
    "A100 PCIe (HBM2e, 1372 GB/s)":  [0.04, 0.08, 0.16, 0.32, 0.63, 0.93, 1.00, 1.00, 0.99],
}
# Session 6-7 : efficacité ASP (gain réel / gain théorique en octets) par volume lu
EFF_VOL = {"Tesla T4": ([4, 32, 100], [0.283, 0.784, 0.852]),
           "A100 PCIe": ([8, 40, 100], [0.222, 0.306, 0.776])}
# Session 9 : H200, fusion vs deux noyaux, par quota q
FUSED_Q = {8: (1.57, 1.71, 15, 15), 16: (0.78, 1.27, 3, 15),
           32: (0.68, 0.71, 0, 15), 64: (0.32, 0.38, 0, 14)}
# Session 10 : bout en bout Qwen3-8B sur H200
E2E = dict(N=[16384, 32768, 65536, 131072, 262144, 524288],
           dense=[29.86, 29.88, 29.76, 29.98, 42.13, 72.62],
           asp=[49.63, 49.71, 49.52, 49.40, 54.01, 89.78],
           ratio=[13.47, 17.66, 20.90, 23.01, 24.24, 24.90])
# Session 10 : ventilation de l'attention seule (x36 couches), après correction GQA
VENT = dict(N=[16384, 65536, 131072, 262144],
            attn_dense=[2.06, 4.06, 6.62, 11.34],
            attn_asp=[20.39, 20.30, 20.93, 23.05],
            floor=[4.96, 6.87, 9.41, 14.50])


def fig_law():
    """Fig. 1 — la loi de sélection : rappel de masse en fonction du bruit."""
    p = ROOT / "12_poc/resultats/law_selection.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    key = "avec RoPE" if "avec RoPE" in d else list(d)[0]
    ref = d[key]["ref"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, mk in (("4", "o"), ("8", "s"), ("16", "^")):
        if k not in ref: continue
        xs = sorted(float(x) for x in ref[k])
        ys = [100 * ref[k][str(x) if str(x) in ref[k] else f"{x}"] for x in xs]
        ax.plot(xs, ys, marker=mk, ms=3, lw=1.2, label=f"$k={k}$")
    meth = d[key]["methods"]
    for nm, mk, col in (("coreset r=8", "*", C["asp"]), ("cobs r=8", "D", C["fused"]),
                        ("mean", "v", C["dense"])):
        if nm in meth:
            ax.scatter([meth[nm]["sigma"]], [100 * meth[nm]["r@8"]], marker=mk, s=45,
                       color=col, zorder=5, edgecolor="white", linewidth=0.5, label=nm)
    ax.set_xscale("log"); ax.set_xlabel(r"estimator noise $\sigma$ (nats)")
    ax.set_ylabel("mass recall (%)"); ax.set_xlim(0.03, 5)
    ax.legend(fontsize=6, ncol=2, loc="lower left", frameon=False)
    fig.savefig(FIG / "law.pdf"); plt.close(fig)


def fig_gather():
    """Fig. 2 — pénalité du gather par granularité, trois architectures mémoire."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for (nm, (x, y)), col in zip(GATHER.items(), [C["igpu"], C["t4"], C["a100"]]):
        a1.plot(x, y, marker="o", ms=3.2, lw=1.3, color=col, label=nm)
        a2.plot(x, [100 * s for s in SAT[nm][:len(x)]], marker="s", ms=3.2, lw=1.3, color=col)
    a1.axhline(1.0, color="k", lw=0.7, ls=":")
    a1.axvline(1024, color="k", lw=0.7, ls="--")
    a1.set_xscale("log", base=2); a1.set_ylim(0.6, 1.12)
    a1.set_xlabel("gathered chunk size (bytes)")
    a1.set_ylabel(r"gather / contiguous bandwidth")
    a1.legend(fontsize=6, loc="lower right", frameon=False)
    a2.axhline(80, color="k", lw=0.7, ls=":")
    a2.set_xscale("log", base=2); a2.set_xlabel("gathered chunk size (bytes)")
    a2.set_ylabel("% of peak bandwidth (contiguous)")
    fig.savefig(FIG / "gather.pdf"); plt.close(fig)


def fig_fusion():
    """Fig. 3 — H200 : la fusion ne gagne qu'à faible quota q."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    qs = sorted(FUSED_Q)
    f2k = [FUSED_Q[q][0] for q in qs]
    gvp = [FUSED_Q[q][1] for q in qs]
    x = np.arange(len(qs)); w = 0.38
    ax.bar(x - w/2, f2k, w, color=C["fused"], label="fused / two-kernel")
    ax.bar(x + w/2, gvp, w, color=C["asp"], label="fused / flat")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    for i, q in enumerate(qs):
        w_, n_ = FUSED_Q[q][2], FUSED_Q[q][3]
        ax.text(i, max(f2k[i], gvp[i]) + 0.06, f"{w_}/{n_}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([f"q={q}" for q in qs])
    ax.set_ylabel(r"speedup ($\times$)"); ax.set_ylim(0, 2.1)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_xlabel("per-stratum quota (winning configurations above)")
    fig.savefig(FIG / "fusion.pdf"); plt.close(fig)


def fig_e2e():
    """Fig. 4 — bout en bout Qwen3-8B : octets économisés vs latence réelle."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    N = np.array(E2E["N"]) / 1000
    a1.plot(N, E2E["dense"], marker="o", ms=3.5, lw=1.3, color=C["dense"], label="dense")
    a1.plot(N, E2E["asp"], marker="s", ms=3.5, lw=1.3, color=C["asp"], label="ASP (unfused)")
    a1.set_xscale("log", base=2); a1.set_xlabel("context (thousands of tokens)")
    a1.set_ylabel("decode-step latency (ms)")
    a1.legend(fontsize=7, frameon=False, loc="upper left")
    a1b = a1.twinx(); a1b.grid(False)
    a1b.plot(N, E2E["ratio"], marker="^", ms=3, lw=1.0, ls=":", color=C["fused"])
    a1b.set_ylabel(r"bytes read: dense / ASP ($\times$)", color=C["fused"], fontsize=8)
    a1b.tick_params(axis="y", labelcolor=C["fused"], labelsize=7)

    Nv = np.array(VENT["N"]) / 1000
    a2.plot(Nv, VENT["attn_dense"], marker="o", ms=3.5, lw=1.3, color=C["dense"],
            label="dense attention")
    a2.plot(Nv, VENT["attn_asp"], marker="s", ms=3.5, lw=1.3, color=C["asp"],
            label="ASP selection (unfused)")
    a2.plot(Nv, VENT["floor"], lw=1.0, ls="--", color="#888888",
            label="memory floor (weights+KV)")
    a2.set_xscale("log", base=2); a2.set_xlabel("context (thousands of tokens)")
    a2.set_ylabel("per-step time, 36 layers (ms)")
    a2.legend(fontsize=6.5, frameon=False, loc="center left")
    fig.savefig(FIG / "e2e.pdf"); plt.close(fig)


def fig_strat():
    """Fig. 5 — coût de la stratification : entrelacée vs contiguë."""
    p = ROOT / "12_poc/resultats/stratified.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    key = "avec RoPE" if "avec RoPE" in d else list(d)[0]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ms = sorted(int(x) for x in d[key])
    for mm, mk in zip(ms, ["o", "s", "^", "v"]):
        e = d[key][str(mm)]
        Gs = sorted(int(g) for g in e["entrelacee"])
        base = e["global_"]
        ax.plot(Gs, [100 * e["entrelacee"][str(g)] / base for g in Gs],
                marker=mk, ms=3, lw=1.2, color=C["asp"], alpha=0.85,
                label=f"interleaved, m={mm}" if mm == ms[0] else None)
        ax.plot(Gs, [100 * e["contigue"][str(g)] / base for g in Gs],
                marker=mk, ms=3, lw=1.2, ls="--", color=C["fused"], alpha=0.85,
                label=f"contiguous, m={mm}" if mm == ms[0] else None)
    ax.set_xscale("log", base=2); ax.set_xlabel("number of strata $G$")
    ax.set_ylabel("recall relative to global top-$m$ (%)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.savefig(FIG / "strat.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_law, fig_gather, fig_fusion, fig_e2e, fig_strat):
        try:
            f(); print(f"  {f.__name__:12s} -> OK")
        except Exception as e:
            print(f"  {f.__name__:12s} -> ECHEC : {type(e).__name__}: {e}")
    print("\nfigures dans", FIG)
