#!/usr/bin/env python3
"""
精确拆分 Prefill vs Decode 吞吐的 Benchmark
使用 stream=True 逐 token 测量
"""
import json, time, statistics, sys, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

API_URL = "http://localhost:8000/v1/completions"
MODEL = "gemma4"
MAX_TOKENS = 512  # 更多 token 以稳定 decode 测量
TEMPERATURE = 0.0

# 不同长度的 prompt 测试
PROMPTS = {
    "short": "Explain how transformers work.",
    "medium": "Explain how transformer attention mechanisms work in large language models, including multi-head attention, self-attention, and cross-attention.",
    "long": """Explain how transformer attention mechanisms work in large language models. Include detailed explanations of multi-head attention, self-attention, cross-attention, the query/key/value projection mechanism, scaling factors, softmax normalization, and how these components work together to enable contextual understanding. Also explain the differences between encoder-decoder attention and decoder-only architectures, and how causal masking works in autoregressive generation. Provide a thorough technical explanation suitable for someone with a solid machine learning background who wants to understand the mathematical foundations of modern LLMs.""",
}

CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]
REQUESTS_PER_LEVEL = {1: 8, 4: 12, 8: 16, 16: 24, 32: 32}


def stream_request(rid, prompt_key, timeout=180):
    """Send streaming request and measure per-token timing."""
    prompt = PROMPTS[prompt_key]
    prompt_tokens = len(prompt.split())  # rough estimate

    payload = json.dumps({
        "model": MODEL, "prompt": prompt,
        "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        API_URL, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    token_times = []
    first_token_time = None
    full_text = ""
    ttft = None
    total_time = None
    error = None

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        start_time = time.time()

        buffer = b""
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or line.startswith(b":"):
                    continue
                if line.startswith(b"data: "):
                    data_str = line[6:].decode("utf-8", errors="replace").strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        now = time.time()
                        if first_token_time is None:
                            first_token_time = now
                            ttft = now - start_time
                        choices = data.get("choices", [])
                        if choices and "text" in choices[0]:
                            delta = choices[0]["text"]
                            full_text += delta

                        token_times.append(now)
                    except json.JSONDecodeError:
                        pass

        end_time = time.time()
        total_time = end_time - start_time

        # Parse usage from last chunk or estimate
        # vLLM streaming returns usage in the final data frame
        generated_tokens = len(full_text.split()) if full_text.strip() else 0

        return {
            "rid": rid,
            "ok": True,
            "prompt_key": prompt_key,
            "prompt_len": len(prompt),
            "ttft": ttft,
            "total_time": total_time,
            "generated_tokens": generated_tokens,
            "num_token_timings": len(token_times),
            "error": None,
        }

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        return {"rid": rid, "ok": False, "error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"rid": rid, "ok": False, "error": str(e)[:150]}


def run_level(c, n_reqs, prompt_key):
    """Run benchmark at given concurrency."""
    print(f"\n{'='*60}")
    print(f"  C{c}  ({n_reqs} requests)  prompt={prompt_key}")
    print(f"{'='*60}")

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=c) as ex:
        futs = {}
        idx = 0
        # Initial batch
        for _ in range(min(c, n_reqs)):
            f = ex.submit(stream_request, idx, prompt_key)
            futs[f] = idx
            idx += 1
        # Process and submit remaining
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            if idx < n_reqs:
                f2 = ex.submit(stream_request, idx, prompt_key)
                futs[f2] = idx
                idx += 1

    wall = time.time() - start

    # Filter successful
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    sr = len(ok) / len(results) * 100 if results else 0

    if not ok:
        print("  ❌ All requests failed!")
        if fail:
            for f in fail[:3]:
                print(f"     {f['error']}")
        return None

    # === Prefill metrics ===
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    ttft_avg = statistics.mean(ttfts) if ttfts else 0

    # Prefill throughput: prompt tokens / TTFT (per-request)
    # Estimate: our medium prompt is ~25 tokens
    prompt_token_counts = {
        "short": 6,
        "medium": 28,
        "long": 98,
    }
    ptokens = prompt_token_counts.get(prompt_key, 20)
    prefill_toks_s = ptokens / ttft_avg if ttft_avg > 0 else 0

    # === Decode metrics ===
    tpots = []
    for r in ok:
        gt = r["generated_tokens"]
        tt = r["ttft"]
        tot = r["total_time"]
        if gt > 0 and tt is not None:
            decode_time = tot - tt
            # First token comes from prefill, so decode_count = gt - 1
            decode_count = max(gt - 1, 1)
            tpot = decode_time / decode_count
            tpots.append(tpot)

    tpot_avg = statistics.mean(tpots) if tpots else 0
    tpot_ms = tpot_avg * 1000

    # Decode throughput (tokens per second during generation phase)
    decode_toks_s = 1.0 / tpot_avg if tpot_avg > 0 else 0

    # === Aggregate (total tokens / total wall time) — like original benchmark ===
    total_gen_tokens = sum(r["generated_tokens"] for r in ok)
    aggregate_tok_s = total_gen_tokens / wall if wall > 0 else 0

    # Per-user average
    per_user_tok_s = aggregate_tok_s / c if c > 0 else 0

    # === Chunked prefill analysis ===
    # If chunked prefill is enabled, prefill might overlap with decode
    # In that case, pure prefill throughput = prompt_tokens / time_to_first_token
    # Aggregate prefill throughput across batch = sum(prompt_tokens) / avg(ttft)

    print(f"  成功率: {sr:.0f}% ({len(ok)}/{len(results)})")
    print(f"  ─────────────────────────────────────")
    print(f"  🔵 PREFILL:")
    print(f"     TTFT (avg):         {ttft_avg*1000:.0f} ms")
    print(f"     Prefill tok/s:      {prefill_toks_s:.1f} tok/s (prompt={ptokens}tok / {ttft_avg:.2f}s)")
    print(f"  🟢 DECODE (生成):")
    print(f"     TPOT (avg):         {tpot_ms:.1f} ms/tok")
    print(f"     Decode tok/s:       {decode_toks_s:.1f} tok/s")
    print(f"  🟡 混合 (prefill+decode):")
    print(f"     Aggregate tok/s:    {aggregate_tok_s:.1f} tok/s (总{len(ok)}req)")
    print(f"     Per-user tok/s:     {per_user_tok_s:.1f} tok/s")
    print(f"     平均生成长度:       {total_gen_tokens/len(ok):.0f} tok/req")

    if fail:
        print(f"  ⚠️  {len(fail)} failed:")
        for f in fail[:2]:
            print(f"     {f['error']}")

    return {
        "concurrency": c,
        "prompt": prompt_key,
        "prompt_tokens": ptokens,
        "success_rate": sr,
        "n_ok": len(ok),
        "n_total": len(results),
        "ttft_ms": ttft_avg * 1000,
        "ttft_s": ttft_avg,
        "prefill_tok_s": prefill_toks_s,
        "tpot_ms": tpot_ms,
        "decode_tok_s": decode_toks_s,
        "aggregate_tok_s": aggregate_tok_s,
        "per_user_tok_s": per_user_tok_s,
        "avg_gen_tokens": total_gen_tokens / len(ok) if ok else 0,
        "total_gen_tokens": total_gen_tokens,
        "wall_time_s": wall,
    }


def main():
    prompt_key = "medium"  # Use medium prompt for fair comparison

    print("=" * 60)
    print("  🔬 Prefill vs Decode 吞吐测量")
    print(f"  model: gemma4 | {MODEL}")
    print(f"  prompt: '{PROMPTS[prompt_key][:60]}...'")
    print(f"  max_tokens: {MAX_TOKENS}")
    print(f"  stream: True (逐 token 计时)")
    print("=" * 60)

    all_results = []
    for c in CONCURRENCY_LEVELS:
        r = run_level(c, REQUESTS_PER_LEVEL[c], prompt_key)
        if r:
            all_results.append(r)
        time.sleep(3)

    # Print summary
    print("\n\n" + "=" * 90)
    print("  📊 预填充 vs 解码 吞吐汇总")
    print("=" * 90)
    print(f"{'并发':>5} | {'TTFT':>8} | {'预填充tok/s':>11} | {'TPOT':>7} | {'解码tok/s':>10} | {'混合tok/s':>10} | {'单用户tok/s':>12}")
    print("-" * 90)

    for r in all_results:
        print(f"{r['concurrency']:>5} | {r['ttft_ms']:>6.0f}ms | {r['prefill_tok_s']:>10.1f} | "
              f"{r['tpot_ms']:>5.0f}ms | {r['decode_tok_s']:>9.1f} | "
              f"{r['aggregate_tok_s']:>9.1f} | {r['per_user_tok_s']:>11.1f}")
    print("-" * 90)
    print("  预填充tok/s = prompt_tokens / TTFT (处理输入的速度)")
    print("  解码tok/s   = 1 / TPOT (生成输出的速度)")
    print("  混合tok/s   = 总生成tokens / 墙钟时间 (prefill+decode混合, 与原文一致)")
    print("=" * 90)

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "config": {
            "prompt": PROMPTS[prompt_key],
            "max_tokens": MAX_TOKENS,
            "streaming": True,
        },
        "results": all_results,
    }
    with open("/home/nvidia/vLLM/prefill_decode_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ 结果已保存: /home/nvidia/vLLM/prefill_decode_results.json")


if __name__ == "__main__":
    main()
