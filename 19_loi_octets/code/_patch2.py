import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code\frontiere2.py")
t = p.read_text(encoding="utf-8")
t = t.replace('    ap.add_argument("--out", default="frontiere.json")',
              '    ap.add_argument("--out", default="frontiere.json")\n    ap.add_argument("--rope", default="both")')
t = t.replace('        for rope in (True, False):',
              '        ropes = (True, False) if a.rope == "both" else (a.rope == "true",)\n        for rope in ropes:')
old = '    (OUT / a.out).write_text(json.dumps(allout, indent=2), encoding="utf-8")'
new = '''    fp = OUT / a.out
    prev = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    prev.update(allout)
    fp.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    allout = prev'''
assert old in t
t = t.replace(old, new)
t = t.replace('    print("->", OUT / a.out,', '    print("->", fp,')
p.write_text(t, encoding="utf-8")
print("patched2 OK")
