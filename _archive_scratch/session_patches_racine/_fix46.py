import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\kappa_T_propre.py")
s = p.read_text(encoding="utf-8")
s = s.replace('MASS = {"n": 0, "s": 0.0}', 'MASS = {4: {"n": 0, "s": 0.0}, 16: {"n": 0, "s": 0.0}}')
old = '''        if Lb is None:
            # masse piegee : clefs de la clef de bloc partielle derriere la fenetre
            Am = A.mean(0)
            pmin = max(PMIN, W + Lb + 1)
            if pmin < qn - 1:
                idx = torch.arange(pmin, qn)
                bstar = ((idx - W + 1) // Lb) - 1
                start = (bstar + 1) * Lb
                pos = torch.arange(qn)[None, :]
                gap = (pos >= start[:, None]) & (pos <= (idx - W)[:, None])
                MASS["s"] += float((Am[pmin:qn] * gap).sum())
                MASS["n"] += len(idx)'''
new = '''        if Lb is None:
            Am = A.mean(0)
            for Lbv in (4, 16):
                pmin = max(PMIN, W + Lbv + 1)
                if pmin < qn - 1:
                    idx = torch.arange(pmin, qn)
                    bstar = ((idx - W + 1) // Lbv) - 1
                    start = (bstar + 1) * Lbv
                    pos = torch.arange(qn)[None, :]
                    gap = (pos >= start[:, None]) & (pos <= (idx - W)[:, None])
                    MASS[Lbv]["s"] += float((Am[pmin:qn] * gap).sum())
                    MASS[Lbv]["n"] += len(idx)'''
assert old in s
s = s.replace(old, new)
old2 = '''    d = run(T, nch, None)
    M = MASS["s"] / max(MASS["n"], 1)
    l4 = run(T, nch, 4); M4 = MASS["s"] / max(MASS["n"], 1)
    MASS["s"] = MASS["n"] = 0
    l16 = run(T, nch, 16); M16 = MASS["s"] / max(MASS["n"], 1)
    MASS["s"] = MASS["n"] = 0'''
new2 = '''    d = run(T, nch, None)
    M4 = MASS[4]["s"] / max(MASS[4]["n"], 1)
    M16 = MASS[16]["s"] / max(MASS[16]["n"], 1)
    for kk in (4, 16):
        MASS[kk]["s"] = MASS[kk]["n"] = 0
    l4 = run(T, nch, 4)
    l16 = run(T, nch, 16)'''
assert old2 in s
s = s.replace(old2, new2)
p.write_text(s, encoding="utf-8")
print("corrige")
