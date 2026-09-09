#!/bin/bash
R=~/.local/share/uv/tools/google-colab-cli/lib/python3.13/site-packages/colab_cli/runtime.py
if [ ! -f "$R.orig" ]; then cp "$R" "$R.orig"; echo "sauvegarde: $R.orig"; fi
python3 - "$R" <<'PY'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
old = "self._kernel_client = jupyter_kernel_client.KernelClient("
new = ("_KC = getattr(jupyter_kernel_client, 'KernelClient', None) or "
       "jupyter_kernel_client.JupyterKernelClient\n"
       "                    self._kernel_client = _KC(")
if old in t:
    t = t.replace(old, new)
    p.write_text(t)
    print("patch applique : alias KernelClient -> JupyterKernelClient")
elif "_KC = getattr(" in t:
    print("patch deja present")
else:
    print("MOTIF INTROUVABLE - rien fait")
PY
echo "=== verification syntaxe ==="
~/.local/share/uv/tools/google-colab-cli/bin/python -c "import ast,sys; ast.parse(open('$R').read()); print('OK')"
