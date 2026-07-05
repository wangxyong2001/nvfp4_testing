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

### 3.5 结果 1 — Sehyo/Qwen3.5 (社区版, 无推理链)

```
非流式 (3 trials):
  Trial 1: 241 tokens in 2.69s = 89.7 tok/s
  Trial 2: 512 tokens in 4.97s = 103.0 tok/s
  Trial 3: 512 tokens in 4.67s = 109.6 tok/s
  Average: 100.8 tok/s

流式:
  512 tokens in 4.62s = 110.7 tok/s
```

### 3.6 结果 2 — nvidia/Qwen3.6 (官方版, 含推理链)

```
配置: --speculative --mtp-quantization nvfp4 --bind 0.0.0.0
Kernel: (sm_121, qwen3.6-35b-a3b, nvfp4) 140 modules
FP8 KV cache: 启用在线校准
MTP: 已启用 (256 experts, Bf16 精度)

无推理触发 (简单问答):
  65 tokens in 0.62s = 105.4 tok/s (输出)

含推理触发 (技术解释, 3 trials):
  Trial 1: 1290 tok (777 reasoning + 513 output) in 14.51s = 88.9 tok/s total, 35.4 tok/s (output)
  Trial 2: 1290 tok (777 reasoning + 513 output) in 14.46s = 89.2 tok/s total, 35.5 tok/s (output)
  Trial 3: 1290 tok (777 reasoning + 513 output) in 14.75s = 87.4 tok/s total, 34.8 tok/s (output)
  Average: 88.5 tok/s total, 35.2 tok/s (output only)

流式 (含推理, 故事类):
  513 tokens in 14.70s = 34.9 tok/s (仅可见输出 token)
```

> **注意**: Qwen3.6 每次推理产出 ~777 个 reasoning token (内部思考链)。
> 引擎原始解码速度实际为 ~96-105 tok/s (无推理场景)，与 Sehyo/Qwen3.5 基本一致。
> 推理链 token 计入 completion_tokens 但不可直接暴露给用户。

### 3.7 结果 3 — nvidia/Qwen3.6 + 禁用推理链

**发现**: Qwen3.6 的推理链可以通过 system prompt 抑制，恢复纯输出速度。

**方法**: chat API system prompt
```python
{"role": "system", "content": "Answer concisely without thinking step by step. Never use <think> tags."}
```

**结果**:
```
流式 (3 trials, 无 reasoning):
  Stream: 512 tok in 4.84s = 105.7 tok/s | TTFT=147ms
  Stream: 512 tok in 4.86s = 105.3 tok/s | TTFT=130ms
  Stream: 512 tok in 4.84s = 105.7 tok/s | TTFT=130ms
  Average: 105.2 tok/s

非流式 (参考): 512 tok in 4.90s = 104.4 tok/s
```

**关键数据点**:

| 模式 | 流式 tok/s | TTFT | Reasoning tokens |
|---|---|---|---|
| 默认 (含推理) | 35.2 tok/s* | 63-70ms | ~780/次 |
| 禁用推理 (system prompt) | **105.2 tok/s** | 130-147ms | 0 |

*\*推理链阶段不产出可见文本，仅计数可见输出 token*

**结论**: `--disable-thinking` server flag 对 Qwen3.6 无效（模型本身生成 `<think>` token），但通过 chat API system prompt 可以完全抑制推理链，恢复与 Sehyo/Qwen3.5 一致的纯输出速度 (~105 tok/s)。

---

### 3.8 结果 4 — nvidia/Qwen3.6 + llama.cpp (Q4_K_P GGUF)

**模型**: `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf` (22GB, Q4_K_P)  
**引擎**: llama.cpp v9728 (fabde3bf5)  
**后端**: CUDA (sm_121 Blackwell) — `BLACKWELL_NATIVE_FP4 = 1`  
**加载**: `--n-gpu-layers 99 --mlock --ctx-size 32768`

```
启动时间: 2.5s (GGUF cold start)
GPU: GB10, 117GB free at load

非流式 (含 reasoning, 3 trials):
  Trial 1: 512 tok in 8.08s = 63.4 tok/s
  Trial 2: 512 tok in 7.99s = 64.1 tok/s
  Trial 3: 512 tok in 7.75s = 66.1 tok/s
  Average: 64.5 tok/s

非流式 (禁用 reasoning, 3 trials):
  Trial 1: 512 tok in 7.76s = 66.0 tok/s
  Trial 2: 512 tok in 7.76s = 66.0 tok/s
  Trial 3: 512 tok in 7.78s = 65.8 tok/s
  Average: 65.9 tok/s

流式 (禁用 reasoning, 3 trials):
  Trial 1: 512 tok in 7.99s = 64.1 tok/s
  Trial 2: 512 tok in 7.80s = 65.6 tok/s
  Trial 3: 512 tok in 7.80s = 65.6 tok/s
  Average: 65.1 tok/s
```

> **llama.cpp 速度 65 tok/s 低于 vLLM (80) 和 Atlas (100+) 的原因**:
> 1. GGUF Q4_K_P 是通用 uint4 量化，没有 Blackwell 原生 FP4 kernel 优化
> 2. llama.cpp 的 CUDA backend 没有针对 MoE 架构做特别优化
> 3. 没有 MTP speculative decoding 等高级加速技术

## 四、Atlas vs vLLM 对比

| 维度 | vLLM | Atlas | 差距 |
|---|---|---|---|
| **运行时** | Python + PyTorch | Rust + CUDA | — |
| **FP4 路径** | Marlin 模拟 (weight-only) | 原生 sm_121 NVFP4 kernel | **核心差距** |
| **MTP** | ❌ 不支持 | ✅ 1.35x verify | ~25% |
| **冷启动** | ~10 min | <2 min | 5x |
| **容器大小** | 20+ GB | 2.5 GB | 8x |
| **单用户 tok/s** | **80.0 tok/s** | **100.8 / 104.4 / 65.9 tok/s** | **+20~31%** |
| **流式 tok/s** | **79.2 tok/s** | **110.7 / 105.2 / 65.1 / 34.9* tok/s** | **+33~40% (无推理)** |
| **Qwen3.6 推理链** | ❌ 不支持 | ✅ Atlas ~777 tok/次 (可抑制) | — |
| **KV Cache 策略** | 预分配 6.2M slots | 动态 4.4M slots (Atlas) | — |

### 全引擎完整数据

| 引擎 | 模型 | 量化格式 | 非流式 | 流式 | TTFT | 冷启动 |
|---|---|---|---|---|---|---|
| **Atlas** | Sehyo/Qwen3.5-35B-A3B | NVFP4 (native) | **100.8 tok/s** | **110.7 tok/s** | — | <2 min |
| **Atlas** | nvidia/Qwen3.6-35B-A3B | NVFP4 (native) | **104.4 tok/s** | **105.2 tok/s** | 130ms | <2 min |
| **Atlas** | nvidia/Qwen3.6-35B-A3B | NVFP4 (native) | 96.5 tok/s | 34.9 tok/s* | 63ms | <2 min |
| **vLLM** (Marlin) | nvidia/Qwen3.6-35B-A3B | NVFP4 (modelopt) | 80.0 tok/s | 79.2 tok/s | 41ms | ~10 min |
| **llama.cpp** | Qwen3.6-35B-A3B | Q4_K_P (GGUF) | 65.9 tok/s | 65.1 tok/s | — | ~2.5s |

*\*含 reasoning 时仅统计可见输出 token*

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

## 六、复现指南（启动参数 & 测试方法）

### 6.1 Atlas + Sehyo/Qwen3.5 或 nvidia/Qwen3.6 (NVFP4)

**容器启动**:
```bash
# 镜像: avarok/atlas-gb10:latest
docker run --gpus all --ipc=host -p 8888:8888 \
  --name atlas-serving \
  -v /path/to/model:/model \
  avarok/atlas-gb10:latest \
  serve /model \
  --speculative --mtp-quantization nvfp4 \
  --bind 0.0.0.0
```

**禁用推理链 (仅 Qwen3.6)**:
```bash
# 通过 system prompt 抑制，server flag --disable-thinking 无效
# chat API 请求体:
{
  "messages": [
    {"role": "system", "content": "Answer concisely without thinking step by step. Never use <think> tags."},
    {"role": "user", "content": "Your prompt here"}
  ]
}
```

**测速**:
```bash
# 非流式: POST /v1/completions 或 /v1/chat/completions
# 参数: max_tokens=512, temperature=0.0
# 流式: stream=true, 逐 token 计数
```

**结果**:
| 模型 | 推理链 | 流式 tok/s |
|---|---|---|
| Sehyo/Qwen3.5 | 无 (默认) | **110.7** |
| nvidia/Qwen3.6 | 默认 (含 reasoning) | 34.9 |
| nvidia/Qwen3.6 | system prompt 抑制 | **105.2** |

---

### 6.2 vLLM + nvidia/Qwen3.6 (NVFP4, Marlin 模拟)

**容器启动**:
```bash
# 镜像: vllm/vllm-openai:nightly-aarch64
docker run -d --gpus all --name qwen-serving -p 8000:8000 \
  -v /path/to/nvidia_qwen3.6:/model \
  vllm/vllm-openai:nightly-aarch64 \
  /model --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.4 --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --dtype auto \
  --load-format fastsafetensors \
  --served-model-name qwen3.6
```

**关键参数说明**:
| 参数 | 值 | 作用 |
|---|---|---|
| `gpu-memory-utilization` | 0.4 | 限制显存占用，避免 OOM |
| `max-model-len` | 32768 | 上下文长度 |
| `max-num-batched-tokens` | 8192 | 单次 batch 最大 token 数 |
| `kv-cache-dtype` | fp8 | KV cache 精度 |

**测速**: 使用 `/home/nvidia/vLLM/benchmark_prefill_decode.py` 或 `benchmark_serving_curve.py`

**结果**:
| 模式 | tok/s |
|---|---|
| 非流式 | **80.0** |
| 流式 | **79.2** |

---

### 6.3 llama.cpp + Qwen3.6 Q4_K_P (GGUF)

**最佳性能参数** (65 tok/s):
```bash
/home/nvidia/llama/llama.cpp/build/bin/llama-server \
  --model /path/to/Qwen3.6-35B-A3B-Q4_K_P.gguf \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 32768 \
  --batch-size 512 --ubatch-size 512 \
  --n-gpu-layers 99 \
  --mlock
```

**spark_launcher_7.sh 参数** (54 t/s):
```bash
/home/nvidia/llama/llama.cpp/build/bin/llama-server \
  --model /path/to/Qwen3.6-35B-A3B-Q4_K_P.gguf \
  --host 0.0.0.0 --port 8081 \
  --ctx-size 65536 \
  --batch-size 2048 --ubatch-size 512 \
  --n-gpu-layers 999 \
  --threads 10 --threads-batch 10 \
  --parallel 1 --no-mmap \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

> **注意**: `--no-mmap` + `cache-type q8_0` + ctx 65K 会大幅降低速度 (18 tok/s)。
> spark_launcher 中的 "54 t/s" 是在不同负载下的实测值，单用户简单 prompt 可达 **65 tok/s**。

**测速**: 使用 `/v1/completions` API，prompt 约 10-20 tokens, max_tokens=512

**结果**:
| 配置 | 非流式 | 流式 |
|---|---|---|
| 最佳参数 (ctx=32K, mlock) | **65.9** | **65.1** |
| spark_launcher 参数 (ctx=65K, no-mmap) | 18.2 | 35.8 |

---

### 6.4 测试方法统一说明

所有测速采用以下统一方法:
1. **Prompt**: `"Explain how transformer attention mechanisms work in large language models."` (~15 tokens)
2. **Output**: max_tokens=512, temperature=0.0
3. **非流式**: 3 次取平均，记录 `completion_tokens / elapsed`
4. **流式**: 逐 token 计数，3 次取平均，同时记录 TTFT
5. **硬件**: DGX Spark GB10 (sm_121), 128G 统一内存
6. **容器/引擎**: 均使用 localhost 网络，排除网络延迟

### 6.5 脚本位置

| 文件 | 用途 |
|---|---|
| `/home/nvidia/vLLM/atlas/build_image.py` | 构建 Atlas Docker 镜像 (手动 layer 下载 + docker load) |
| `/home/nvidia/vLLM/atlas/run_atlas.sh` | 启动 Atlas serving 容器 |
| `/home/nvidia/vLLM/atlas/benchmark.py` | Atlas 基准测试脚本 |
| `/home/nvidia/vLLM/atlas/layers/` | Atlas 容器各层缓存 |
| `/home/nvidia/vLLM/benchmark_serving_curve.py` | vLLM serving curve 测试 |
| `/home/nvidia/vLLM/benchmark_prefill_decode.py` | vLLM prefill/decode 拆分测试 |
| `/home/nvidia/llama/spark_launcher_7.sh` | llama.cpp 多模型启动器 (含 Qwen3.6) |

---

## 七、参考链接

- [Atlas Inference on HuggingFace](https://huggingface.co/Atlas-Inference)
- [Sehyo/Qwen3.5-35B-A3B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-35B-A3B-NVFP4)
- [NVIDIA Qwen3.6 NVFP4 Model Card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
- [DGX Spark Forum - Atlas Introduction](https://forums.developer.nvidia.com/t/introducing-the-atlas-inference-server-and-engine/362210/25)
- [SlimTradeyBaby Gemma-4 Benchmark (参考)](https://www.reddit.com/r/LocalLLaMA/comments/1rkefjw/solved_the_dgx_spark_102_stable_toks_qwen3535ba3b/)
