#!/bin/bash
V=~/.local/share/uv/tools/google-colab-cli
echo "=== dependances declarees par google-colab-cli ==="
$V/bin/python - <<'PY'
import importlib.metadata as md
try:
    d = md.distribution("google-colab-cli")
    print("version colab-cli:", d.version)
    for r in (d.requires or []):
        if "kernel" in r.lower() or "jupyter" in r.lower():
            print("  requiert:", r)
except Exception as e:
    print("err", e)
PY
echo "=== versions disponibles de jupyter-kernel-client ==="
$V/bin/python -m pip index versions jupyter-kernel-client 2>&1 | head -4
echo "=== usage exact dans runtime.py ==="
sed -n '95,120p' $V/lib/python3.13/site-packages/colab_cli/runtime.py
