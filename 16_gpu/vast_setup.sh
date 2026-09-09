#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
vastai set api-key "$1" >/dev/null 2>&1
chmod 600 ~/.config/vastai/vast_api_key 2>/dev/null || chmod 600 ~/.vast_api_key 2>/dev/null
echo "=== instance ==="
vastai show instance 49255728 --raw 2>&1 | python3 -c "
import json,sys
d=json.load(sys.stdin)
if isinstance(d,dict) and d.get('error'): print('ERREUR:',d); sys.exit()
for k in ('id','actual_status','gpu_name','num_gpus','gpu_ram','cuda_max_good','ssh_host','ssh_port','public_ipaddr','direct_port_start','direct_port_end','machine_dir_ssh_port','image_uuid','disk_space','cpu_name'):
    if k in d: print(f'  {k:22s} {d[k]}')
" 2>&1 | head -30
