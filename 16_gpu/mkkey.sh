#!/bin/bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
F=~/.ssh/asp_h200
if [ ! -f $F ]; then
  ssh-keygen -t ed25519 -N "" -C "asp-paper-h200" -f $F >/dev/null 2>&1
  echo "nouvelle cle generee : $F"
else
  echo "cle deja presente : $F"
fi
chmod 600 $F
echo "--- cles disponibles ---"
ls -1 ~/.ssh/*.pub 2>/dev/null
echo "--- CLE PUBLIQUE A AJOUTER ---"
cat $F.pub
