#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
for C in "-p 9006 root@50.46.253.92" "-p 20815 root@ssh8.vast.ai"; do
  if timeout 30 ssh $O $C "echo ok" 2>/dev/null | grep -q ok; then echo "$C" > /tmp/h200conn; break; fi
done
[ -f /tmp/h200conn ] || { echo "ECHEC ssh"; exit 1; }
C=$(cat /tmp/h200conn); echo "connexion: $C"
ssh $O $C "nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; nvcc --version|tail -2|head -1; nproc"
