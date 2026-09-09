#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
echo "=== commandes utiles ==="
vastai --help 2>&1 | grep -iE "execute|logs|ssh|copy|attach|scp" | head -20
echo "=== cles ssh du compte ==="
vastai show ssh-keys 2>&1 | head -10
echo "=== cle locale ? ==="
ls -la ~/.ssh/*.pub 2>/dev/null || echo "(aucune cle publique locale)"
