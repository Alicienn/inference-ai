import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code\frontiere2.py")
t = p.read_text(encoding="utf-8")
old = """def lse(A, axis):
    M = A.max(axis=axis, keepdims=True)
    return (M + np.log(np.exp(A - M).sum(axis=axis, keepdims=True)))[..., 0]"""
new = """def lse(A, axis):
    M = A.max(axis=axis, keepdims=True)
    R = M + np.log(np.exp(A - M).sum(axis=axis, keepdims=True))
    return np.squeeze(R, axis=axis)"""
assert old in t, "lse introuvable"
t = t.replace(old, new)
old2 = '                Tm = np.stack([S.true_logmass(Ks[i], Q, s) for i in range(len(cand))], 1)'
new2 = '                Tm = lse(s * np.einsum("nld,jd->nlj", Ks, Q), 1)'
assert old2 in t, "true_logmass introuvable"
t = t.replace(old2, new2)
p.write_text(t, encoding="utf-8")
print("patched OK")
