#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
command -v vastai >/dev/null 2>&1 || { echo "installation vastai..."; uv tool install vastai >/dev/null 2>&1 || pip install --quiet --break-system-packages vastai; }
export PATH="$HOME/.local/bin:$PATH"
echo "=== version ==="; vastai --version 2>&1 | head -2
echo "=== instance 49255728 ==="
vastai show instance 49255728 --api-key "$VAST_KEY" --raw 2>&1 | head -100
