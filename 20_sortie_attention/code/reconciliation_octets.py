# -*- coding: utf-8 -*-
"""Reconciliation des deux comptabilites d'octets d'ASP.
Harness e2e (16_gpu/e2e_qwen.py) : etage fin = re-classement sur les RESUMES C2 (m*r2 vecteurs
de segment), aucune lecture de cle candidate brute ; budget k=n/kfrac blocs.
Banc qualite (20_sortie_attention/code/bilan_octets.py) : etage fin = lecture des cles BRUTES
des 2m candidats ; budget W + m*Lb cles.
Unite : 1 element pleine precision (D dims) = 1 cle-equivalent.
"""
import json, pathlib

ROOT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
OUT = ROOT / "20_sortie_attention" / "resultats" / "reconciliation_octets.txt"
J = json.loads((ROOT / "16_gpu" / "resultats" / "e2e_qwen_rtx4090.json").read_text(encoding="utf-8"))
CFG = J["cfg"]; L, KVH, D = J["L"], J["KVH"], J["D"]
lines = [f"HARNESS {J['model']} : L={L} KVH={KVH} D={D} cfg={CFG}",
         "unite = 1 cle-equivalent (D elements pleine precision) par requete/tete/couche", "",
         f"{'N':>8s} {'n':>6s} {'m':>5s} {'k':>4s} | {'dense':>9s} {'ASP':>9s} {'ratio calc':>10s} "
         f"{'ratio enreg.':>12s} | {'% dense':>8s}",
         "-" * 92]
for row in J["rows"]:
    N = row["N"]; n = N // CFG["Lb"]; m = max(n // CFG["mfrac"], 1); k = max(n // CFG["kfrac"], 1)
    rd = 2.0 * N
    c1 = n * CFG["r1"]; c2 = m * CFG["r2"]
    sel = 2.0 * (k * CFG["Lb"] + CFG["nloc"] + CFG["nsink"])
    ra = c1 + c2 + sel
    lines.append(f"{N:8d} {n:6d} {m:5d} {k:4d} | {rd:9.0f} {ra:9.0f} {rd/ra:10.2f} "
                 f"{row['byte_ratio']:12.2f} | {100*ra/rd:8.2f}")
lines += ["", "DECOMPOSITION DE L'ECART AVEC LE BANC QUALITE (T=1024, W=128, m=32, Lb=4) :",
          f"  banc qualite, un etage          : 2W + 2mLb = {2*128 + 2*32*4} unites sur 2T = {2*1024} -> "
          f"{100*(2*128+2*32*4)/(2*1024):.1f} % de dense, soit {2*1024/(2*128+2*32*4):.2f}x",
          f"  banc qualite, deux etages exact : 2W + 3mLb = {2*128 + 3*32*4} unites -> "
          f"{100*(2*128+3*32*4)/(2*1024):.1f} % de dense, soit {2*1024/(2*128+3*32*4):.2f}x",
          f"  banc qualite, deux etages 8 bits: 2W + 2mLb = {2*128 + 2*32*4} unites -> "
          f"{100*(2*128+2*32*4)/(2*1024):.1f} % de dense, soit {2*1024/(2*128+2*32*4):.2f}x",
          "",
          "  Terme 1 (budget de blocs) : le harness lit k*Lb = n/64 blocs, soit "
          f"{100.0/(CFG['kfrac']):.2f} % des cles, contre 25 % pour le banc qualite.",
          "  Terme 2 (mecanisme de l'etage fin) : le harness re-classe sur C2 = m*r2 vecteurs de",
          f"    segment ({CFG['r2']} par bloc) et ne lit AUCUNE cle candidate brute ; le banc qualite",
          "    lit les 2m*Lb cles brutes des candidats. A N=131072 : C2 = "
          f"{max((131072//CFG['Lb'])//CFG['mfrac'],1)*CFG['r2']} unites contre 2m*Lb = "
          f"{2*max((131072//CFG['Lb'])//CFG['mfrac'],1)*CFG['Lb']} unites, soit un facteur "
          f"{2*max((131072//CFG['Lb'])//CFG['mfrac'],1)*CFG['Lb'] / (max((131072//CFG['Lb'])//CFG['mfrac'],1)*CFG['r2']):.0f}.",
          "",
          "CONCLUSION : les deux chiffres ne sont pas contradictoires, ils decrivent deux algorithmes",
          "differents. Le ratio 13,5-23x du harness vient d'un etage fin a RESUMES (budget 1,6 % des",
          "cles) ; le ratio 3,9x du banc qualite vient d'un etage fin a CLES BRUTES (budget 25 %).",
          "Seul le second a ete valide en qualite dans cette session ; les proxys valides du premier",
          "(resumes fins) sont les configurations cosinus 32d et max 32d, qui coutaient +0,069 a",
          "+0,410 nat contre +0,011 nat pour le reclassement exact."]
OUT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
