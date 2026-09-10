import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\index_appris_v2.py")
t = p.read_text(encoding="utf-8")
old = '''def orth(Praw):
    Qm, _ = torch.linalg.qr(Praw.transpose(1, 2))     # (NL,H,D,dp)
    return Qm.transpose(1, 2)                          # (NL,H,dp,D)'''
new = '''def orth(Praw):
    Qm, _ = torch.linalg.qr(Praw.transpose(2, 3))     # (NL,H,D,dp) : QR sur (D,dp)
    return Qm.transpose(2, 3)                          # (NL,H,dp,D) : lignes orthonormees'''
assert old in t
p.write_text(t.replace(old, new), encoding="utf-8")
print("QR corrige")
