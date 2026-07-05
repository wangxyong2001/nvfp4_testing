# DGX Spark NVFP4 推理引擎基准测试笔记

> 日期: 2026-07-06
> 硬件: NVIDIA DGX Spark (GB10 Blackwell, sm_120/sm_121, 128G 统一内存)
> 模型族: Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (MoE 256E→A8, 35B 总参/3B 激活)

---

## 一、测试背景

### 1.1 目标

在 DGX Spark GB10 上测试 NVIDIA NVFP4 量化模型的推理速度，探索突破 100 tok/s 单用户输出速度的可能性。

### 1.2 模型来源

| 模型 | 来源 | 量化 | 大小 |
|---|---|---|---|
| nvidia/Qwen3.6-35B-A3B-NVFP4 | NVIDIA 官方 | NVFP4 (W4A16) | ~18GB |
| Sehyo/Qwen3.5-35B-A3B-NVFP4 | Sehyo (社区) | NVFP4 | ~24GB |

### 1.3 硬件约束

GB10 (sm_120/sm_121) **没有原生 FP4 Tensor Core 支持**。vLLM 走 Marlin 模拟路径，Atlas 使用自研原生 NVFP4 kernel。

---

## 二、引擎一：vLLM (Marlin 模拟路径)

### 2.1 环境

- 引擎: vLLM v0.23.1rc1.dev471 (nightly-aarch64)
- 容器: vllm/vllm-openai:nightly-aarch64
- 运行时: Python 3.12 + PyTorch 2.12 + CUDA 13.0
- 冷启动: ~10 分钟

### 2.2 启动命令

```bash
docker run -d --gpus all --name qwen-serving -p 8000:8000 \
  -v /home/nvidia/vLLM/nvidia_qwen3.6:/model \
  vllm/vllm-openai:nightly-aarch64 \
  /model --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.4 --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --dtype auto \
  --load-format fastsafetensors \
  --served-model-name qwen3.6
```

### 2.3 测试配置矩阵

| 配置 | gpu-mem | ctx | batch | chunked | prefix | 结果 |
|---|---|---|---|---|---|---|
| A (最佳) | 0.4 | 32K | 8192 | ❌ | ❌ | **80.0 tok/s** |
| B | 0.85 | 65K | 16384 | ❌ | ❌ | 73.3 tok/s |
| C (HF官方) | 0.85 | 262K | 8192 | ✅ | ✅ | 70.3 tok/s |

### 2.4 详细结果 (配置 A)

```
非流式: 80.0 tok/s (256-2048 tokens, 线性扩展)
流式解码: 79.2 tok/s (TPOT=12.5ms, TTFT=41ms)
3 次测试一致性: 79.8-80.2 tok/s (<1% 波动)
```

### 2.5 vLLM 失败尝试

- **MTP Speculative Decoding**: ❌ `--speculative-model` 等 flag 在 nightly-aarch64 不受支持
- **num-scheduler-steps 8**: ❌ 该版本不支持
- **更高 GPU 内存 (0.85) + 更长上下文 (65K)**: ❌ 速度反而降低

### 2.6 vLLM 瓶颈分析

1. **Marlin Kernel 模拟路径**: 无原生 FP4 → weight-only 4-bit 模拟，计算效率低
2. **Python/PyTorch 调度开销**: 每次推理经过 Python GIL → PyTorch dispatcher → CUDA
3. **MoE 路由负担**: 256 专家门控计算

---

## 三、引擎二：Atlas (原生 NVFP4 引擎)

### 3.1 什么是 Atlas

Atlas 是**纯 Rust + CUDA 推理引擎**，专为 DGX Spark GB10 (sm_121) 编写：
- 零 Python/零 PyTorch 依赖
- ~2.5GB Docker 镜像 (vs vLLM 20+GB)
- 冷启动 <2 分钟
- 手调 sm_121 CUDA kernel

### 3.2 安装与部署

#### 下载容器镜像 (Docker Hub API 直连)

由于 Docker daemon 网络限制，无法直接 `docker pull`，改为手动下载 14 层 layer：

```bash
# 步骤1: 获取 auth token
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:avarok/atlas-gb10:pull" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# 步骤2: 获取 manifest 确定 layers
curl -s -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://registry-1.docker.io/v2/avarok/atlas-gb10/manifests/latest"

# 步骤3: 下载每层 blob → tar.gz
# 步骤4: 打包 tar → docker load
```

见 `/home/nvidia/vLLM/atlas/build_image.py`

#### 模型下载

```bash
# 使用 Python huggingface_hub，通过 hf-mirror.com 加速
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('Sehyo/Qwen3.5-35B-A3B-NVFP4')
"
```

模型放置: `/home/nvidia/vLLM/Sehyo_Qwen3.5-35B-A3B-NVFP4/`
权重: 22G (model.safetensors) + 1.6G (extra_weights.safetensors) = ~24GB

### 3.3 启动命令

```bash
docker run --gpus all --ipc=host -p 8888:8888 \
  --name atlas-serving \
  -v /home/nvidia/vLLM/Sehyo_Qwen3.5-35B-A3B-NVFP4:/model \
  avarok/atlas-gb10:latest \
  serve /model \
  --speculative --mtp-quantization nvfp4 \
  --bind 0.0.0.0
```

### 3.4 启动日志关键信息

```
Kernel target: (sm_121, qwen3.6-35b-a3b, nvfp4) (140 modules)
Model: 40 layers, 10 attention, 30 SSM, 256 experts
KV cache: 4.4M max tokens
Fast weight loader (O_DIRECT + pipelined): 23.31 GB → 30.30 GB peak → ~66GB free
MTP Speculative: ENABLED (1.35x verify multiplier)
```

### 3.5 基准测试结果

```
非流式 (3 trials):
  Trial 1: 241 tokens in 2.69s = 89.7 tok/s
  Trial 2: 512 tokens in 4.97s = 103.0 tok/s
  Trial 3: 512 tokens in 4.67s = 109.6 tok/s
  Average: 100.8 tok/s

流式:
  512 tokens in 4.62s = 110.7 tok/s
```

---

## 四、Atlas vs vLLM 对比

| 维度 | vLLM | Atlas | 差距 |
|---|---|---|---|
| **运行时** | Python + PyTorch | Rust + CUDA | — |
| **FP4 路径** | Marlin 模拟 (weight-only) | 原生 sm_121 NVFP4 kernel | **核心差距** |
| **MTP** | ❌ 不支持 | ✅ 1.35x verify | ~25% |
| **冷启动** | ~10 min | <2 min | 5x |
| **容器大小** | 20+ GB | 2.5 GB | 8x |
| **单用户 tok/s** | **80.0 tok/s** | **100.8 tok/s** | **+26%** |
| **流式 tok/s** | **79.2 tok/s** | **110.7 tok/s** | **+40%** |
| **KV Cache 策略** | 预分配 6.2M slots | 动态 4.4M slots | — |

### 为什么 Atlas 更快

1. **原生 CUDA kernel**: 为 GB10 sm_121 手写 NVFP4 kernel，不走 Marlin 模拟路径（预计贡献 +20-30%）
2. **零 Python 开销**: Rust 直接调 CUDA，没有 PyTorch dispatcher 和 Python GIL（预计贡献 +10-15%）
3. **MTP 流水线**: MTP speculative decoding 每次验证多生成 1.35 个 token（贡献 +25%）
4. **Kernel Fusion**: FlashAttention-2/4 + SageAttention 3 + LeanAttention 融合（贡献 +15-20%）

---

## 五、关键问题

### 5.1 原生 FP4 到底重不重要

**非常重要。** 同样的模型（Qwen3.5/3.6-35B），vLLM 走 Marlin (weight-only 模拟) 只能跑 80 tok/s，而 Atlas 的 sm_121 原生 NVFP4 kernel 跑到 100+ tok/s。如果未来 TRT-LLM 支持 Blackwell 原生 FP4 Tensor Core，预期可到 160+ tok/s。

### 5.2 MTP 在不同引擎的支援度

| 特性 | vLLM nightly-aarch64 | Atlas |
|---|---|---|
| MTP flag | ❌ not supported | ✅ native |
| Speculative | ❌ | ✅ 1 draft/step |
| Verify pipeline | ❌ | ✅ pipelined |

### 5.3 Atlas 的限制

- **长上下文**: SSM prefill 在 >10K tokens 时比 vLLM 慢（SSM 状态缓存瓶颈）
- **社区模型兼容性**: 仅支持 Qwen3.5/3.6/Gemma-4 等 ~13 个模型家族
- **新引擎**: 仍在快速迭代，API 可能变化

---

## 六、测试脚本位置

| 文件 | 用途 |
|---|---|
| `/home/nvidia/vLLM/atlas/build_image.py` | 构建 Atlas Docker 镜像 (手动 layer 下载 + docker load) |
| `/home/nvidia/vLLM/atlas/run_atlas.sh` | 启动 Atlas serving 容器 |
| `/home/nvidia/vLLM/atlas/benchmark.py` | Atlas 基准测试脚本 |
| `/home/nvidia/vLLM/atlas/layers/` | Atlas 容器各层缓存 |
| `/home/nvidia/vLLM/benchmark_serving_curve.py` | vLLM serving curve 测试 |
| `/home/nvidia/vLLM/benchmark_prefill_decode.py` | vLLM prefill/decode 拆分测试 |

---

## 七、参考链接

- [Atlas Inference on HuggingFace](https://huggingface.co/Atlas-Inference)
- [Sehyo/Qwen3.5-35B-A3B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4)
- [NVIDIA Qwen3.6 NVFP4 Model Card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
- [DGX Spark Forum - Atlas Introduction](https://forums.developer.nvidia.com/t/introducing-the-atlas-inference-server-and-engine/362210/25)
- [SlimTradeyBaby Gemma-4 Benchmark (参考)](https://www.reddit.com/r/LocalLLaMA/comments/1rkefjw/solved_the_dgx_spark_102_stable_toks_qwen3535ba3b/)
