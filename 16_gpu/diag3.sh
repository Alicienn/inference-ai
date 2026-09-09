#!/bin/bash
V=~/.local/share/uv/tools/google-colab-cli
echo "=== signature de JupyterKernelClient (1.0.2) ==="
$V/bin/python - <<'PY'
import inspect, jupyter_kernel_client as j
C = j.JupyterKernelClient
try:
    print(inspect.signature(C.__init__))
except Exception as e:
    print("sig err:", e)
print("a start():", hasattr(C, "start"))
print("a _own_kernel dans __init__:", "_own_kernel" in (inspect.getsource(C.__init__) if hasattr(C,'__init__') else ""))
PY
echo "=== versions disponibles ==="
uv pip index versions jupyter-kernel-client 2>&1 | head -5 || pip index versions jupyter-kernel-client 2>&1 | head -5
