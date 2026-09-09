#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
ssh $O $C "df -h / /root 2>/dev/null | head -4; echo ---; nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader; echo ---; python3 --version; pip list 2>/dev/null | grep -iE '^(torch|vllm|transformers|flash|accelerate) ' ; echo ---; free -g | head -2"
