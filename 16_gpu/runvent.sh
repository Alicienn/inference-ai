#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
ssh $O $C "cat > /root/vent.py" < /mnt/c/Users/alici/Downloads/inference_opti/16_gpu/vent.py
ssh $O $C "cd /root && python3 vent.py 2>&1 | grep -viE 'warning|it/s'"
