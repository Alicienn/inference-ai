import pathlib
p = pathlib.Path(r"C:\Users\alici\Downloads\inference_opti\19_loi_octets\code\frontiere2.py")
t = p.read_text(encoding="utf-8")
if "--heads" not in t:
    t = t.replace('    ap.add_argument("--rope", default="both")',
                  '    ap.add_argument("--rope", default="both")\n    ap.add_argument("--heads", type=int, default=1)')
    t = t.replace('def run(tag, Lb=64, nq=24, local=8, use_rope=True, seed=0, max_cfg=None):',
                  'def run(tag, Lb=64, nq=24, local=8, use_rope=True, seed=0, max_cfg=None, hstride=1):')
    t = t.replace('            for h in range(kv * g, (kv + 1) * g):',
                  '            for h in range(kv * g, (kv + 1) * g, hstride):')
    t = t.replace('res, rad = run(tag, Lb=a.block, nq=a.nq, use_rope=rope, max_cfg=a.maxcfg)',
                  'res, rad = run(tag, Lb=a.block, nq=a.nq, use_rope=rope, max_cfg=a.maxcfg, hstride=a.heads)')
    p.write_text(t, encoding="utf-8")
    print("patched5 OK")
else:
    print("deja patche")
