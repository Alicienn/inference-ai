"""Recherche robuste dans le corpus texte (gere les octets nuls / encodages)."""
import re, sys, pathlib

TXT = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\_txt")

def load(p):
    raw = p.read_bytes().replace(b"\x00", b"")
    return raw.decode("utf-8", errors="replace")

def search(pattern, filt=None, ctx=0, limit=40):
    rx = re.compile(pattern, re.I)
    n = 0
    for p in sorted(TXT.glob("*.txt")):
        if filt and filt.lower() not in p.stem.lower():
            continue
        lines = load(p).splitlines()
        for i, ln in enumerate(lines):
            if rx.search(ln):
                n += 1
                if n > limit:
                    print("... (tronque)")
                    return
                lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
                for j in range(lo, hi):
                    print(f"{p.stem[:38]:38s} {j+1:5d} | {lines[j][:190]}")
                if ctx:
                    print("-")

if __name__ == "__main__":
    pat = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else None
    ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    search(pat, filt, ctx, lim)
