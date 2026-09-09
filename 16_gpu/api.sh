#!/bin/bash
K=~/.ssh/asp_h200
O="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o LogLevel=ERROR"
C="-p 9006 root@50.46.253.92"
ssh $O $C 'python3 - <<PY
import transformers, inspect
print("transformers", transformers.__version__)
from transformers.models.qwen3 import modeling_qwen3 as M
print("classes:", [x for x in dir(M) if "Attention" in x or "attention" in x][:8])
src = inspect.getsource(M.Qwen3Attention.forward)
print("--- signature forward ---")
print(src[:1200])
PY'
