#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
until ssh $O $C "python3 -c \"
from huggingface_hub import snapshot_download
p=snapshot_download('Qwen/Qwen3-8B', local_files_only=True)
import glob,os
f=glob.glob(p+'/*.safetensors')
print('PRET', len(f), 'fichiers', round(sum(os.path.getsize(x) for x in f)/2**30,1),'Go')
\" 2>/dev/null" | grep -q PRET; do sleep 20; done
ssh $O $C "python3 -c \"
from huggingface_hub import snapshot_download
import glob,os
p=snapshot_download('Qwen/Qwen3-8B', local_files_only=True)
f=glob.glob(p+'/*.safetensors')
print('modele pret :', len(f),'fichiers,', round(sum(os.path.getsize(x) for x in f)/2**30,1),'Go')\""
df_out=$(ssh $O $C "df -h / | tail -1"); echo "disque: $df_out"
