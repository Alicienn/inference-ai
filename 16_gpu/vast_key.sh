#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ ! -f ~/.ssh/vast_ed25519 ]; then
  ssh-keygen -t ed25519 -N "" -C "asp-bench-tmp" -f ~/.ssh/vast_ed25519 >/dev/null
  echo "cle generee"
else
  echo "cle deja presente"
fi
PUB=$(cat ~/.ssh/vast_ed25519.pub)
echo "=== enregistrement sur le compte ==="
vastai create ssh-key "$PUB" 2>&1 | head -3
echo "=== attachement a l'instance ==="
vastai attach ssh 49255728 "$PUB" 2>&1 | head -3
echo "=== url ssh ==="
vastai ssh-url 49255728 2>&1 | head -3
