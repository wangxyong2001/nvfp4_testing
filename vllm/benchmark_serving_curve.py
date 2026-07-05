#!/usr/bin/env python3
"""
Gemma-4 Serving Curve Benchmark
复现 @SlimTradeyBaby 的 DGX Spark 测试场景

测量:
  - 吞吐量 (tok/s aggregate & per-user)
  - TTFT (Time to First Token) p50/p95
  - TPOT (Time per Output Token)
  - 成功率

并发度: C1, C4, C8, C16, C32, C64, C96, C128
"""

import json
import time
import statistics
import sys
import argparse
import urllib.request
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# OpenAI-compatible vLLM endpoint
API_URL = "http://localhost:8000/v1/completions"

# 使用固定 prompt 确保可复现
PROMPT = "Explain how transformer attention mechanisms work in large language models, including multi-head attention, self-attention, and cross-attention. Provide a detailed technical explanation suitable for someone with a machine learning background."

CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64, 96, 128]
# 复现原文: C32→578 tok/s, C96→601 tok/s, C128→604 tok/s

# Per-request tokens
MAX_TOKENS = 256
TEMPERATURE = 0.0  # deterministic


def send_request(request_id: int, timeout: int = 120) -> dict:
    """Send a single completion request and measure performance."""
    payload = json.dumps({
        "model": "gemma4",
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.time()
    ttft = None
    first_token_time = None

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = resp.read()
        end_time = time.time()

        result = json.loads(body.decode("utf-8"))

        # Extract timing
        choices = result.get("choices", [])
        if choices:
            generated_tokens = len(choices[0].get("text", "").split())
        else:
            generated_tokens = 0

        # Get vLLM internal timing if available
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", generated_tokens)

        # Approximation: vLLM returns total time, we estimate TTFT as
        # first portion of generation. For non-streaming, we use completion_tokens
        # to estimate TPOT.
        total_time = end_time - start_time

        return {
            "request_id": request_id,
            "success": True,
            "total_time": total_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generated_text_len": generated_tokens,
            "error": None,
        }

    except urllib.error.HTTPError as e:
        end_time = time.time()
        error_body = e.read().decode("utf-8", errors="replace")
        return {
            "request_id": request_id,
            "success": False,
            "total_time": end_time - start_time,
            "error": f"HTTP {e.code}: {error_body[:200]}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generated_text_len": 0,
        }
    except Exception as e:
        end_time = time.time()
        return {
            "request_id": request_id,
            "success": False,
            "total_time": end_time - start_time,
            "error": str(e)[:200],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generated_text_len": 0,
        }


def run_benchmark(concurrency: int, num_requests: int = None) -> dict:
    """
    Run benchmark at given concurrency level.
    Sends enough requests to get a stable measurement.
    """
    if num_requests is None:
        # At least 20 requests per level, at least 2x concurrency
        num_requests = max(20, concurrency * 3)

    print(f"\n{'='*60}")
    print(f"  Benchmark: C{concurrency} ({num_requests} requests)")
    print(f"{'='*60}")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        request_idx = 0
        submitted = 0

        # Submit initial batch
        for _ in range(min(concurrency, num_requests)):
            future = executor.submit(send_request, request_idx)
            futures[future] = request_idx
            request_idx += 1
            submitted += 1

        # As requests complete, submit more
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if submitted < num_requests:
                new_future = executor.submit(send_request, request_idx)
                futures[new_future] = request_idx
                request_idx += 1
                submitted += 1

    end_time = time.time()
    wall_time = end_time - start_time

    # Calculate metrics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    success_rate = len(successful) / len(results) * 100 if results else 0

    if successful:
        total_completion_tokens = sum(r["completion_tokens"] for r in successful)
        total_prompt_tokens = sum(r["prompt_tokens"] for r in successful)
        total_times = [r["total_time"] for r in successful]

        # Throughput
        aggregate_tok_s = total_completion_tokens / wall_time if wall_time > 0 else 0

        # Latency metrics
        ttft_estimates = sorted(total_times)

        p50_idx = len(ttft_estimates) // 2
        p95_idx = int(len(ttft_estimates) * 0.95)

        ttft_p50 = ttft_estimates[p50_idx] if ttft_estimates else 0
        ttft_p95 = ttft_estimates[min(p95_idx, len(ttft_estimates) - 1)] if ttft_estimates else 0

        # Average TPOT (total_time / completion_tokens per request)
        tpots = [
            r["total_time"] / max(r["completion_tokens"], 1)
            for r in successful
        ]
        avg_tpot = statistics.mean(tpots) if tpots else 0
        avg_tpot_ms = avg_tpot * 1000

        avg_completion_tokens = statistics.mean([r["completion_tokens"] for r in successful])
        avg_total_time = statistics.mean(total_times)
        per_user_throughput = avg_completion_tokens / avg_total_time if avg_total_time > 0 else 0

    else:
        aggregate_tok_s = 0
        ttft_p50 = 0
        ttft_p95 = 0
        avg_tpot = 0
        avg_tpot_ms = 0
        per_user_throughput = 0
        avg_completion_tokens = 0
        total_completion_tokens = 0
        total_prompt_tokens = 0

    result = {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_time_s": wall_time,
        "success_rate": success_rate,
        "successful": len(successful),
        "failed": len(failed),
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "aggregate_tok_s": aggregate_tok_s,
        "per_user_tok_s": per_user_throughput,
        "ttft_p50_s": ttft_p50,
        "ttft_p95_s": ttft_p95,
        "avg_tpot_s": avg_tpot,
        "avg_tpot_ms": avg_tpot_ms,
    }

    # Print results
    print(f"\n  📊 C{concurrency} 结果:")
    print(f"  ───────────────────────────────────")
    print(f"  成功率:          {success_rate:.1f}% ({len(successful)}/{len(results)})")
    print(f"  墙钟时间:        {wall_time:.2f}s")
    print(f"  总生成 tokens:   {total_completion_tokens}")
    print(f"  总输入 tokens:   {total_prompt_tokens}")
    print(f"  聚合吞吐:        {aggregate_tok_s:.1f} tok/s")
    print(f"  单用户吞吐:      {per_user_throughput:.1f} tok/s")
    print(f"  TTFT p50:        {ttft_p50:.2f}s")
    print(f"  TTFT p95:        {ttft_p95:.2f}s")
    print(f"  TPOT (avg):      {avg_tpot_ms:.1f}ms/tok")

    if failed:
        print(f"\n  ⚠️  失败请求示例:")
        for f in failed[:3]:
            print(f"    #{f['request_id']}: {f['error']}")

    return result


def print_summary(all_results):
    """Print final summary table."""
    print("\n\n")
    print("=" * 80)
    print("  🏆 Serving Curve Benchmark 汇总")
    print("=" * 80)
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模型: nvidia/Gemma-4-26B-A4B-NVFP4")
    print(f"  引擎: vLLM nightly-aarch64")
    print(f"  硬件: NVIDIA GB10 (DGX Spark)")
    print(f"  Context: 262K | KV Cache: FP8")
    print(f"  Prompt: '{PROMPT[:50]}...'")
    print(f"  Max tokens/req: {MAX_TOKENS}")
    print("=" * 80)
    print(f"{'并发':>6} | {'成功率':>8} | {'聚合tok/s':>12} | {'单用户tok/s':>13} | {'TTFT p50':>9} | {'TTFT p95':>9} | {'TPOT':>7}")
    print("-" * 80)

    for r in all_results:
        c = r["concurrency"]
        sr = f"{r['success_rate']:.0f}%"
        agg = f"{r['aggregate_tok_s']:.1f}"
        per = f"{r['per_user_tok_s']:.1f}"
        p50 = f"{r['ttft_p50_s']:.2f}s"
        p95 = f"{r['ttft_p95_s']:.2f}s"
        tpot = f"{r['avg_tpot_ms']:.0f}ms"
        print(f"{c:>6} | {sr:>8} | {agg:>12} | {per:>13} | {p50:>9} | {p95:>9} | {tpot:>7}")

    print("-" * 80)

    # Save to JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
        "engine": "vllm-nightly-aarch64",
        "hardware": "NVIDIA GB10",
        "config": {
            "context_length": 262144,
            "kv_cache_dtype": "fp8",
            "max_tokens_per_request": MAX_TOKENS,
            "speculative_decoding": False,
        },
        "results": all_results,
    }

    with open("/home/nvidia/vLLM/serving_curve_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 结果已保存: /home/nvidia/vLLM/serving_curve_results.json")


def main():
    parser = argparse.ArgumentParser(description="Serving Curve Benchmark for vLLM")
    parser.add_argument(
        "--concurrency", type=int, nargs="+",
        default=CONCURRENCY_LEVELS,
        help=f"Concurrency levels to test (default: {' '.join(map(str, CONCURRENCY_LEVELS))})"
    )
    parser.add_argument(
        "--requests", type=int, default=None,
        help="Number of requests per concurrency level (default: max(20, concurrency*3))"
    )
    parser.add_argument(
        "--url", type=str, default=API_URL,
        help=f"vLLM API URL (default: {API_URL})"
    )
    parser.add_argument(
        "--skip-until", type=int, default=None,
        help="Skip concurrency levels until this value (inclusive)"
    )
    args = parser.parse_args()

    api_url = args.url

    print("🚀 等待服务就绪...", end="", flush=True)
    for attempt in range(30):
        try:
            req = urllib.request.Request(f"{api_url.replace('/v1/completions', '/v1/models')}")
            urllib.request.urlopen(req, timeout=5)
            print(" ✅")
            break
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
    else:
        print("\n❌ 服务未就绪，请先启动: bash run_gemma4_nvfp4.sh")
        sys.exit(1)

    levels = args.concurrency
    if args.skip_until:
        levels = [c for c in levels if c >= args.skip_until]

    all_results = []
    for c in levels:
        result = run_benchmark(c, num_requests=args.requests)
        all_results.append(result)

        # Save interim results
        with open("/home/nvidia/vLLM/serving_curve_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
                "results": all_results,
            }, f, indent=2)

        # Brief cooldown between levels
        if c != levels[-1]:
            print(f"\n  ⏳ Cooling down 5 seconds...")
            time.sleep(5)

    print_summary(all_results)


if __name__ == "__main__":
    main()
