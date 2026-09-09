"""
MODÈLE ANALYTIQUE DU SEUIL DE GRANULARITÉ DU GATHER — posé AVANT la mesure sur HBM.

Objectif : savoir à l'avance pourquoi le seuil de 1 Kio mesuré sur gfx1152 bougerait
(ou non) sur un GPU datacenter, plutôt que de constater après coup.

=======================================================================================
DÉRIVATION
=======================================================================================
Le noyau de gather affecte UN work-group par morceau de C octets. Le noyau contigu, lui,
balaie de grands segments (16 Kio) par work-group. L'asymétrie n'est donc pas dans le
motif d'adresses mais dans le PARALLÉLISME MÉMOIRE (MLP) que chaque schéma peut exposer.

Loi de Little appliquée au sous-système mémoire : pour saturer une bande passante B avec
une latence mémoire lambda, il faut maintenir en vol

    F* = B * lambda   octets

Or les octets en vol du noyau de gather valent

    F = N_res * min(C, U)

  - N_res : nombre de work-groups résidents simultanément (occupation x nb d'unités)
  - U     : profondeur de pipeline atteignable DANS un work-group, en octets. Bornée par
            les registres et le déroulage de boucle ; ~4 a 8 Kio en pratique.

Deux régimes en découlent :

  (a) C >= U : les octets en vol valent N_res*U, INDÉPENDAMMENT de C.
      Le gather expose alors autant de MLP que le flux contigu -> aucune pénalité.

  (b) C <  U : les octets en vol valent N_res*C. La saturation exige

          C >= B*lambda / N_res  =:  C_little

Le seuil observé est donc

    C* = min( U , alpha * B*lambda/N_res )

alpha etant un facteur de recouvrement imparfait, calibré sur la mesure gfx1152.

CE QUE CE MODÈLE PRÉDIT, ET C'EST CONTRE-INTUITIF. On s'attendrait à ce que le seuil
MONTE sur HBM puisque B est 40x plus grand. Mais N_res monte aussi (132 SM x 32 blocs
contre 8 CU x 16). Le seuil varie donc comme B*lambda/N_res, dont les deux facteurs se
compensent largement. Prédiction : le seuil reste du même ordre, voire baisse sur les
petits GPU très parallèles (T4).

SECOND EFFET, ADDITIF : la granularité DRAM. Un accès dispersé de C octets force
l'activation d'une ligne DRAM (row) de taille R. Si C << R, le coût d'activation
(t_RC) n'est amorti que sur C octets. Cet effet pousse le seuil vers le HAUT, et R est
plus grand sur HBM (~1 Kio par pseudo-canal) que sur LPDDR5 (~0,5-1 Kio). Il est
toutefois largement absorbé par le parallélisme de bancs dès qu'il y a assez de
requêtes en vol — donc il se ramène, lui aussi, à la condition de Little.

=======================================================================================
FALSIFICATION
=======================================================================================
Le modèle est faux si, sur T4 ou H100, l'efficacite s'effondre a des tailles de morceau
OU le modele predit la saturation (C >= C*). Dans ce cas l'explication ne serait pas le
MLP mais la structure DRAM, et il faudrait remplacer B*lambda/N_res par un terme en
taille de ligne DRAM.
"""
import numpy as np

# --------------------------------------------------------------------------------
# Fiches matériel. Les latences sont des ordres de grandeur publiés/mesurés
# couramment ; elles sont le paramètre le plus incertain du modèle.
# --------------------------------------------------------------------------------
GPUS = {
    # nom            B (o/s)   lambda(s)  unites  blocs/unite  ligne DRAM (o)
    "gfx1152 (860M, LPDDR5)": dict(B=80e9,   lam=500e-9, units=8,   bpu=16, row=1024),
    "T4 (GDDR6)":             dict(B=320e9,  lam=400e-9, units=40,  bpu=32, row=1024),
    "A100 (HBM2e)":           dict(B=1935e9, lam=550e-9, units=108, bpu=32, row=1024),
    "H100 (HBM3)":            dict(B=3350e9, lam=600e-9, units=132, bpu=32, row=1024),
    "MI300X (HBM3)":          dict(B=5300e9, lam=650e-9, units=304, bpu=16, row=1024),
}
U_BYTES = 6144.0        # profondeur de pipeline intra-groupe, en octets (~6 Kio)


def predict(g, alpha):
    N_res = g["units"] * g["bpu"]
    C_little = alpha * g["B"] * g["lam"] / N_res
    return min(U_BYTES, C_little), C_little, N_res, g["B"] * g["lam"]


def calibrate():
    """
    Calibration de alpha sur la mesure gfx1152 : la bascule y est observee entre
    512 o (ratio 0.83) et 2048 o (ratio 1.01), le point median etant ~1024 o.
    """
    g = GPUS["gfx1152 (860M, LPDDR5)"]
    N_res = g["units"] * g["bpu"]
    C_obs = 1024.0
    return C_obs * N_res / (g["B"] * g["lam"])


if __name__ == "__main__":
    alpha = calibrate()
    print("=" * 92)
    print("MODÈLE DE SEUIL DE GRANULARITÉ DU GATHER — prédictions posées AVANT mesure")
    print("=" * 92)
    print(f"  U (pipeline intra-groupe)  = {U_BYTES:.0f} o")
    print(f"  alpha calibré sur gfx1152  = {alpha:.2f}   (seuil observé 1024 o)")
    print(f"  formule : C* = min(U, alpha * B*lambda / N_res)\n")
    print(f"{'GPU':26s} {'B (GB/s)':>9s} {'F*=B.lam':>10s} {'N_res':>7s} "
          f"{'C_little':>9s} {'SEUIL C*':>9s} {'r2*D*2>=1024 ?':>15s}")
    print("-" * 92)
    for nm, g in GPUS.items():
        Cs, Cl, N, F = predict(g, alpha)
        ok = "OK" if Cs <= 1024 else f"il faudrait {int(np.ceil(Cs/256))*256} o"
        print(f"{nm:26s} {g['B']/1e9:9.0f} {F/1e6:9.2f} Mo {N:7d} "
              f"{Cl:8.0f} o {Cs:8.0f} o {ok:>15s}")

    print("""
LECTURE.
  - F* = B*lambda, les octets a maintenir en vol, explose sur HBM (2,0 Mo sur H100
    contre 0,04 Mo sur le 860M, facteur 50).
  - MAIS N_res explose aussi (4224 blocs sur H100 contre 128), si bien que le seuil
    par morceau C_little reste du meme ordre de grandeur.
  - PREDICTION PRINCIPALE, NUANCEE : la regle r2*D*2 >= 1024 o tient sur T4 (328 o)
    et A100 (1009 o), mais PAS sur H100 (1559 o) ni MI300X (2321 o), ou il faudrait
    la relever a ~2 Kio. Autrement dit le seuil ne s'effondre pas sur HBM, mais il
    monte d'un facteur ~1,5 a 2,3 : la regle doit devenir  r2*D*2 >= 2048 o  pour
    etre sure sur tout le parc datacenter.
  - PREDICTION SECONDAIRE, plus discriminante : sur T4 le seuil devrait etre PLUS BAS
    que sur le 860M (plus de blocs residents pour une bande passante 4x seulement).
    Le gather y sera donc encore plus benin. C'est un test net du modele : si la
    mesure T4 montre au contraire une penalite PLUS FORTE qu'sur le 860M a 512 o,
    le modele MLP est faux et il faut basculer sur une explication DRAM.

SENSIBILITE. Le parametre le plus incertain est lambda. Si la latence HBM3 reelle
est 2x celle supposee (1200 ns au lieu de 600), le seuil H100 double :""")
    g = dict(GPUS["H100 (HBM3)"]); g["lam"] = 1200e-9
    Cs, Cl, N, F = predict(g, alpha)
    print(f"  H100 avec lambda=1200 ns -> C* = {Cs:.0f} o "
          f"({'toujours <= 1024 o' if Cs <= 1024 else 'AU-DESSUS de 1024 o : la regle devrait etre relevee'})")
