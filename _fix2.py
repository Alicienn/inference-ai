import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\20_sortie_attention\code\estimateurs.py")
t = p.read_text(encoding="utf-8")
fixes = [
    ("cen = (kb * wb[..., None]).sum(2) / wb.sum(2).clamp(min=1e-9)",
     "cen = (kb * wb[..., None]).sum(2) / wb.sum(2).clamp(min=1e-9)[..., None]"),
    ("sc = kn[None].expand(H, qn, nb)",
     "sc = kn[:, None, :].expand(H, qn, nb)"),
]
for old, new in fixes:
    assert old in t, f"motif absent: {old[:40]}"
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("2 corrections appliquees")
