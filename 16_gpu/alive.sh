#!/bin/bash
K=~/.ssh/vast_ed25519
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR"
timeout 30 ssh $O -p 13317 root@202.122.49.242 "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; nvcc --version|tail -1; nproc; free -g|head -2" 2>&1 | head -8
