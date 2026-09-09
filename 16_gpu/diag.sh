#!/bin/bash
V=~/.local/share/uv/tools/google-colab-cli
echo "=== version jupyter_kernel_client ==="
$V/bin/python - <<'PY'
import jupyter_kernel_client as j
print("version:", getattr(j, "__version__", "?"))
print("attrs:", [a for a in dir(j) if not a.startswith("_")][:25])
PY
echo "=== paquets ==="
$V/bin/python -m pip list 2>/dev/null | grep -iE "jupyter|kernel|colab"
echo "=== ce que colab-cli attend ==="
grep -rn "jupyter_kernel_client" $V/lib/python3.13/site-packages/colab_cli/runtime.py | head -5
