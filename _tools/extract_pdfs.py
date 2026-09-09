import pypdf, pathlib, sys, re
base = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti")
out = base/"_txt"
for pdf in sorted(base.rglob("*.pdf")):
    dst = out/(pdf.stem + ".txt")
    if dst.exists(): continue
    try:
        r = pypdf.PdfReader(str(pdf))
        t = "\n".join((p.extract_text() or "") for p in r.pages)
        dst.write_text(t, encoding="utf-8")
        print(f"{pdf.stem}: {len(r.pages)}p {len(t)}c")
    except Exception as e:
        print(f"FAIL {pdf.stem}: {e}")
