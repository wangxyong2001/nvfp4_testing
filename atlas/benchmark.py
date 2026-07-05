#!/usr/bin/env python3
"""Atlas benchmark: single-user token/s test"""
import json, time, urllib.request, statistics

API = "http://localhost:8888/v1/completions"
MODEL = "/model"
PROMPT = "Explain how transformer attention mechanisms work in large language models."
MAX_TOKENS = 512
TEMPERATURE = 0.0

def benchmark():
    # Non-streaming test
    payload = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }).encode()
    
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    
    trials = []
    for i in range(3):
        start = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
        elapsed = time.time() - start
        
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        if not tokens:
            tokens = len(result.get("choices", [{}])[0].get("text", "").split())
        
        tok_s = tokens / elapsed if elapsed > 0 else 0
        trials.append(tok_s)
        print(f"  Trial {i+1}: {tokens} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")
    
    print(f"\n  Average: {statistics.mean(trials):.1f} tok/s")
    
    # Streaming test
    print("\n  Streaming test...")
    payload2 = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }).encode()
    
    req2 = urllib.request.Request(API, data=payload2, headers={"Content-Type": "application/json"}, method="POST")
    start = time.time()
    with urllib.request.urlopen(req2, timeout=300) as resp:
        data = resp.read()
    elapsed = time.time() - start
    
    # Count tokens from stream
    tokens = 0
    for line in data.decode().strip().split("\n"):
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                chunk = json.loads(line[6:])
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    tokens += 1
            except:
                pass
    
    tok_s = tokens / elapsed if elapsed > 0 else 0
    print(f"  Stream: {tokens} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")

if __name__ == "__main__":
    print("=" * 50)
    print("  Atlas Benchmark - Sehyo/Qwen3.5-35B-A3B-NVFP4")
    print(f"  DGX Spark GB10 | {MAX_TOKENS} tokens")
    print("=" * 50)
    benchmark()
