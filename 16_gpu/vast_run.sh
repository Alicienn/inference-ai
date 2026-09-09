#!/bin/bash
K=~/.ssh/vast_ed25519
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
C="-p 13317 root@202.122.49.242"
SRC=/mnt/c/Users/alici/Downloads/inference_opti/16_gpu/asp_bench.cpp
echo "=== envoi de la source ($(wc -l < $SRC) lignes) ==="
ssh $O $C "cat > /root/asp_bench.cu" < $SRC && ssh $O $C "wc -l /root/asp_bench.cu"
echo "=== compilation (sm_80) ==="
ssh $O $C "cd /root && nvcc -O3 -arch=sm_80 asp_bench.cu -o asp_bench && echo COMPIL_OK" 2>&1 | tail -5
echo "=== execution ==="
ssh $O $C "cd /root && ./asp_bench" 2>&1
