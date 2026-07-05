#!/bin/bash
# Atlas Inference Server for Qwen3.5-35B-A3B-NVFP4
# DGX Spark GB10 | sm_121 | NVFP4
docker rm -f atlas-serving 2>/dev/null
docker run --gpus all --ipc=host -p 8888:8888 \
  --name atlas-serving \
  -v /home/nvidia/vLLM/Sehyo_Qwen3.5-35B-A3B-NVFP4:/model \
  avarok/atlas-gb10:latest \
  serve /model \
  --speculative --mtp-quantization nvfp4 \
  --bind 0.0.0.0
