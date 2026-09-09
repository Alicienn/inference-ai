import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\balayage_Lb_m.py")
t = p.read_text(encoding="utf-8")
t = t.replace('"balayage_Lb_m.txt"', '"balayage_Lb_m_2.txt"')
old = '''ids = ALL[0:T][None]


def loss():
    with torch.no_grad():
        out = model(ids)
        return float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:]))


CFG.update({"mode": "dense"})
ld = loss()
with log.open("a", encoding="utf-8") as f:
    f.write(f"dense {ld:.4f}\\n\\n")
    f.write("Lb   m   maxip    pca_d8   ecart     frac_cles_lues\\n")
    f.flush()
for Lb in [16, 32, 64]:
    for M in [1, 2, 4]:
        CFG.update({"mode": "sparse", "sel": "maxip", "Lb": Lb, "m": M})
        lm = loss()
        CFG.update({"sel": "pca"})
        lp = loss()
        frac = (M * Lb + W) / T
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{Lb:3d} {M:3d} {lm:8.4f} {lp:8.4f} {lp - lm:+8.4f}  {frac:6.3f}\\n")
            f.flush()
        print(f"Lb={Lb} m={M} maxip={lm:.4f} pca={lp:.4f} ecart={lp-lm:+.4f} frac={frac:.3f}", flush=True)
print("FIN")'''
new = '''def loss(ids):
    with torch.no_grad():
        out = model(ids)
        return float(F.cross_entropy(out.logits[0, :-1], ids[0, 1:]))


for off in [200000, 400000]:
    ids = ALL[off:off + T][None]
    with log.open("a", encoding="utf-8") as f:
        CFG.update({"mode": "dense"})
        f.write(f"\\n=== offset {off} === dense {loss(ids):.4f}\\n")
        f.write("Lb   m   maxip    pca_d8   ecart     frac_cles_lues\\n")
        f.flush()
    for Lb in [16, 32, 64]:
        for M in [1, 2, 4]:
            CFG.update({"mode": "sparse", "sel": "maxip", "Lb": Lb, "m": M})
            lm = loss(ids)
            CFG.update({"sel": "pca"})
            lp = loss(ids)
            frac = (M * Lb + W) / T
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{Lb:3d} {M:3d} {lm:8.4f} {lp:8.4f} {lp - lm:+8.4f}  {frac:6.3f}\\n")
                f.flush()
            print(f"off={off} Lb={Lb} m={M} ecart={lp-lm:+.4f}", flush=True)
print("FIN")'''
assert old in t
p.write_text(t.replace(old, new), encoding="utf-8")
print("balayage etendu a 2 tranches")
