# -*- coding: utf-8 -*-
"""Bilan octets-vs-qualite de l'index ASP a deux etages.
Unite = un element pleine precision (cle ou valeur de dimension D), soit D*2 octets en bf16.
Par requete, par tete, par couche :
  dense          : 2T                      (K et V de toutes les cles)
  un etage       : 2W + 2*m*Lb + idx       (fenetre locale K,V + blocs retenus K,V)
  deux etages p  : 2W + 2*m*Lb*p + m*Lb + idx   (fenetre K,V + K des 2m candidats a la precision p + V des m retenus)
  idx            : nb*d'/(2*D)  (index de blocs a 8 bits)
Les Delta de perte sont ceux mesures dans resultats/echelle_two.txt et gpt2_two.txt.
"""
import pathlib

D = 128          # dimension de tete (SmolLM2)
DP = 8           # rang de projection de l'index
TXT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\resultats\bilan_octets.txt")

# (nom, T, W, m, Lb, pertes mesurees dict)
POINTS = [
    ("GPT-2 T=256 induction", 256, 32, 16, 4,
     {"dense": 0.06211, "un_etage": 0.28764, "deux_exact": 0.07339, "deux_q8": 0.24115}),
    ("Qwen2.5-0.5B T=1024 off 0 (D=64)", 1024, 128, 32, 4,
     {"dense": 2.59795, "un_etage": 2.60562, "deux_exact": 2.59867, "deux_q8": 2.59999, "deux_q4": 2.59811}),
    ("SmolLM2 T=1024 off 0", 1024, 128, 32, 4,
     {"dense": 2.75635, "un_etage": 2.81544, "deux_exact": 2.76814, "deux_q8": 2.76764}),
    ("SmolLM2 T=1024 off 200k", 1024, 128, 32, 4,
     {"dense": 1.63241, "un_etage": 1.64588, "deux_exact": 1.62744, "deux_q8": 1.62737, "deux_q4": 1.62789}),
    ("SmolLM2 T=2048 off 0", 2048, 256, 64, 4,
     {"dense": 2.55973, "un_etage": 2.56925, "deux_exact": 2.56543, "deux_q8": 2.56556, "deux_q4": 2.56584}),
]


def idx_cost(T, Lb):
    nb = T // Lb
    return nb * DP / (2.0 * D)


def costs(T, W, m, Lb):
    i = idx_cost(T, Lb)
    c = {}
    c["dense"] = 2.0 * T
    c["un_etage"] = 2.0 * W + 2.0 * m * Lb + i
    c["deux_exact"] = 2.0 * W + 2.0 * m * Lb + m * Lb + i
    c["deux_q8"] = 2.0 * W + 1.0 * m * Lb + m * Lb + i
    c["deux_q4"] = 2.0 * W + 0.5 * m * Lb + m * Lb + i
    return c


lines = ["BILAN OCTETS-VS-QUALITE DE L'INDEX ASP A DEUX ETAGES",
         f"unite = 1 element pleine precision (D={D}, bf16) ; index d'={DP} a 8 bits ;",
         "octets par requete, par tete, par couche (K et V comptes separement)", ""]
for nom, T, W, m, Lb, pertes in POINTS:
    c = costs(T, W, m, Lb)
    dense = pertes["dense"]
    lines.append(f"=== {nom}  (T={T}, W={W}, m={m}, Lb={Lb}, {100.0*(W+m*Lb)/T:.0f} % de cles a 1 etage) ===")
    lines.append(f"{'config':12s} {'unites':>8s} {'x dense':>8s} {'x 1 etage':>10s} {'perte':>9s} {'Delta':>9s}")
    base = c["un_etage"]
    for k in ("dense", "un_etage", "deux_exact", "deux_q8", "deux_q4"):
        if k not in pertes:
            continue
        lines.append(f"{k:12s} {c[k]:8.1f} {c[k]/c['dense']:8.3f} {c[k]/base:10.3f} "
                     f"{pertes[k]:9.5f} {pertes[k]-dense:+9.5f}")
    lines.append(f"  fenetre locale : {2*W:.0f} unites ; K des 2m candidats : {2*m*Lb:.0f} ; "
                 f"V des m retenus : {m*Lb:.0f} ; index : {idx_cost(T,Lb):.1f}")
    lines.append("")
TXT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
