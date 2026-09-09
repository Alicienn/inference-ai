#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
for f in e2e_qwen.json vent.json; do
  ssh $O $C "cat /root/results/$f" > /mnt/c/Users/alici/Downloads/inference_opti/16_gpu/resultats/h200_$f 2>/dev/null && echo "recupere: h200_$f"
done
