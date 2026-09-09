#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd /mnt/c/Users/alici/Downloads/inference_opti/16_gpu
exec colab run --gpu T4 colab_driver.py 2>&1
