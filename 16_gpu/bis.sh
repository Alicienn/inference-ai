#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
ssh $O $C "cat > /root/bisect.cu" < /mnt/c/Users/alici/Downloads/inference_opti/16_gpu/bisect.cu
ssh $O $C "cd /root && nvcc -O3 -arch=sm_90 bisect.cu -o bisect 2>&1|tail -6 && ./bisect"
