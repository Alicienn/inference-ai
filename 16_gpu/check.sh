#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
ssh $O $C "python3 -c 'import torch;print(\"torch\",torch.__version__)' 2>&1|tail -1; du -sh ~/.cache/huggingface 2>/dev/null; df -h / | tail -1"
