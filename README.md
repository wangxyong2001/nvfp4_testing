# NVFP4 Testing on DGX Spark (GB10)

> DGX Spark GB10 (sm_121 Blackwell, 128G 统一内存) 上测试 NVFP4 量化模型推理速度的完整记录。

## 测试结果摘要

| 引擎 | 模型 | 非流式 | 流式 | 说明 |
|---|---|---|---|---|
| vLLM (Marlin 模拟) | nvidia/Qwen3.6-35B-A3B-NVFP4 | **80.0 tok/s** | 79.2 tok/s | vLLM baseline |
| Atlas (原生 NVFP4) | nvidia/Qwen3.6-35B-A3B-NVFP4 | **96.5 tok/s** | 34.9 tok/s* | 含 ~777 reasoning tok |
| Atlas (原生 NVFP4) | nvidia/Qwen3.6-35B-A3B-NVFP4 | **104.4 tok/s** | **105.2 tok/s** | 🆕 禁用 reasoning |
| Atlas (原生 NVFP4) | Sehyo/Qwen3.5-35B-A3B-NVFP4 | **100.8 tok/s** | **110.7 tok/s** | 无推理链，纯输出 |

*\*流式仅统计可见输出 token，推理链阶段不产出文本*

**Atlas 在 DGX Spark 上突破 100 tok/s 目标，相比 vLLM 提升 +20~40%。**
**Qwen3.6 reasoning 可通过 system prompt 抑制，恢复纯输出速度 105 tok/s。**

## 目录结构

```
nvfp4_testing/
├── atlas/                    # Atlas 推理引擎测试
│   ├── build_image.py        # Docker 镜像构建 (Docker Hub API 直连)
│   ├── run_atlas.sh          # Atlas serving 启动脚本
│   └── benchmark.py          # 基准测试脚本
├── vllm/                     # vLLM 推理引擎测试
│   ├── benchmark_prefill_decode.py   # prefill/decode 拆分测试
│   └── benchmark_serving_curve.py    # serving curve 测试
├── docs/
│   └── GB10-NVFP4-Benchmark-Notes-2026-07-06.md   # 完整测试笔记
└── README.md
```

## 引擎对比

| 维度 | vLLM | Atlas |
|---|---|---|
| 运行时 | Python + PyTorch | Rust + CUDA |
| FP4 路径 | Marlin 模拟 (weight-only) | 原生 sm_121 NVFP4 kernel |
| MTP Speculative | ❌ 不支持 | ✅ 1.35x verify |
| 冷启动 | ~10 min | <2 min |
| 容器大小 | 20+ GB | 2.5 GB |
| 容器加速 | vllm/vllm-openai:nightly-aarch64 | avarok/atlas-gb10:latest |

## 硬件环境

- **设备**: NVIDIA DGX Spark (GB10 Blackwell)
- **CUDA Arch**: sm_120 / sm_121
- **内存**: 128G 统一内存
- **OS**: Linux (定制 NVIDIA 内核)

## 参考链接

- [Atlas Inference on HuggingFace](https://huggingface.co/Atlas-Inference)
- [Sehyo/Qwen3.5-35B-A3B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4)
- [NVIDIA Qwen3.6 NVFP4 Model Card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
- [DGX Spark Forum - Atlas Introduction](https://forums.developer.nvidia.com/t/introducing-the-atlas-inference-server-and-engine/362210/25)
