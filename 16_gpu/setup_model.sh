#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
ssh $O $C 'bash -s' <<'REMOTE'
set -e
export DEBIAN_FRONTEND=noninteractive
echo "=== pip install (torch cu126 + transformers) ==="
pip install --quiet --break-system-packages torch --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2
pip install --quiet --break-system-packages transformers accelerate hf_transfer 2>&1 | tail -2
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo "=== telechargement Qwen3-8B ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.safetensors","*.json","*.txt"])
print("MODELE:", p)
PY
df -h / | tail -1
REMOTE
