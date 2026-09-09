#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
echo "=== 1) vastai execute ==="
timeout 90 vastai execute 49255728 "nvidia-smi --query-gpu=name,memory.total --format=csv" 2>&1 | head -8
echo
echo "=== 2) methodes d'auth ssh proposees ==="
timeout 25 ssh -p 13317 -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 \
    -o PreferredAuthentications=none root@202.122.49.242 true 2>&1 | head -6
echo
echo "=== 3) avec la cle generee ==="
timeout 25 ssh -i ~/.ssh/vast_ed25519 -p 13317 -o StrictHostKeyChecking=no -o BatchMode=yes \
    -o ConnectTimeout=15 root@202.122.49.242 "nvidia-smi --query-gpu=name --format=csv,noheader" 2>&1 | head -4
