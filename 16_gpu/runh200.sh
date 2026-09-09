#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
D=/mnt/c/Users/alici/Downloads/inference_opti/16_gpu
ssh $O $C "cat > /root/fused.cu" < $D/fused.cu
ssh $O $C "cd /root && nvcc -O3 -arch=sm_90 fused.cu -o fused 2>&1 | tail -12 && echo COMPIL_OK"
ssh $O $C "cd /root && ./fused 64 9" 2>&1
