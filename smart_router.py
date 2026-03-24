"""
Smart Model Router + Streaming for LM Studio
─────────────────────────────────────────────
Routes queries to the optimal model based on complexity.
Supports streaming for faster time-to-first-token.

Models (fastest → highest quality):
  Qwen3-0.6B       → simple questions, quick facts    (~60 tok/s)
  LFM 2.5-VL-1.6B  → medium queries, summaries        (~35 tok/s)
  Qwen2.5-Coder-7B → complex coding, multi-step tools (~12 tok/s)

Usage:
  py smart_router.py --prompt "What is 2+2?"          # auto-routes
  py smart_router.py --prompt "Write a Python class"  # routes to 7B
  py smart_router.py --stream "Tell me about Hawaii"  # streaming mode
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

LOG_DIR = Path(r"C:\path\to\your\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("router")

BASE_URL = "http://<IP>:<PORT>"

# ── Model Tiers ──────────────────────────────────────────────
MODELS = {
    "fast": {
        "id": "qwen3-0.6b",
        "load_cmd": "lmstudio-community/Qwen3-0.6B-GGUF",
        "speed": "~60 tok/s",
        "size_mb": 805,
    },
    "balanced": {
        "id": "lfm2.5-vl-1.6b",
        "load_cmd": "lmstudio-community/LFM2.5-VL-1.6B-GGUF",
        "speed": "~35 tok/s",
        "size_mb": 1580,
    },
    "quality": {
        "id": "qwen2.5-coder-7b-instruct",
        "load_cmd": "qwen2.5-coder-7b-instruct",
        "speed": "~12 tok/s",
        "size_mb": 4680,
    },
}


# ═══════════════════════════════════════════════════════════════
#  Complexity Classifier
# ═══════════════════════════════════════════════════════════════

# Anchors for Semantic Routing
FAST_ANCHOR = "what who when where date time weather short define true false fact quick simple calculations"
BALANCED_ANCHOR = "summarize describe list outline explain middle document lookup search context"
QUALITY_ANCHOR = "write code implement design architecture python sas javascript sql api logic debug complex workflow pipeline multi-step rationale"

def classify_complexity(query: str) -> str:
    """Classify query using zero-shot semantic routing via embeddings."""
    try:
        import numpy as np
        payload = json.dumps({
            "model": "text-embedding-nomic-embed-text-v1.5",
            "input": [query, FAST_ANCHOR, BALANCED_ANCHOR, QUALITY_ANCHOR]
        }).encode()
        
        req = urllib.request.Request(
            f"{BASE_URL}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        
        vecs = [np.array(e["embedding"]) for e in resp["data"]]
        q_vec = vecs[0]
        
        def cos_sim(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
        scores = {
            "fast": cos_sim(q_vec, vecs[1]),
            "balanced": cos_sim(q_vec, vecs[2]),
            "quality": cos_sim(q_vec, vecs[3])
        }
        
        # Length acts as a minor heuristic weight multiplier
        word_count = len(query.split())
        if word_count > 40:
            scores["quality"] += 0.15
        elif word_count < 8:
            scores["fast"] += 0.15
            
        best = max(scores, key=scores.get)
        return best
        
    except Exception as e:
        log.warning("Semantic routing failed (fallback to length rules): %s", e)
        words = len(query.split())
        if words <= 10: return "fast"
        elif words <= 30: return "balanced"
        return "quality"


# ═══════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════

def get_loaded_model() -> Optional[str]:
    """Get the currently loaded model ID."""
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/models")
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        models = [m["id"] for m in resp.get("data", [])]
        return models[0] if models else None
    except Exception:
        return None


def is_model_loaded(tier: str) -> bool:
    """Check if the model for the given tier is loaded."""
    current = get_loaded_model()
    if not current:
        return False
    target = MODELS[tier]["id"]
    return target in current.lower() or current.lower() in target


# ═══════════════════════════════════════════════════════════════
#  Streaming Response
# ═══════════════════════════════════════════════════════════════

def stream_response(prompt: str, model: str = "auto", max_tokens: int = 500) -> Generator[str, None, None]:
    """Stream a response token by token. Yields text chunks."""
    payload = json.dumps({
        "model": model if model != "auto" else get_loaded_model() or "qwen2.5-coder-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    resp = urllib.request.urlopen(req, timeout=120)

    buffer = b""
    for chunk in iter(lambda: resp.read(64), b""):
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                try:
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue


def stream_print(prompt: str, model: str = "auto"):
    """Stream and print a response in real-time."""
    print(f"\n  Model: {get_loaded_model()}")
    print(f"  Streaming: ", end="", flush=True)

    t0 = time.time()
    total_chars = 0
    for chunk in stream_response(prompt, model):
        print(chunk, end="", flush=True)
        total_chars += len(chunk)

    elapsed = time.time() - t0
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        est_tokens = len(tokenizer.encode(prompt + " " * int(total_chars)))  # Accurate prompt+response estimation
        # Alternatively, just estimate completion tokens roughly since stream chunks are bytes
        est_tokens = int(total_chars / 3.5) 
    except ImportError:
        est_tokens = int(total_chars / 3.5)

    tok_s = est_tokens / elapsed if elapsed > 0 else 0
    print(f"\n\n  [{est_tokens} tokens, ~{tok_s:.1f} tok/s, {elapsed:.1f}s]")


# ═══════════════════════════════════════════════════════════════
#  Smart Router
# ═══════════════════════════════════════════════════════════════

def route_and_respond(prompt: str, stream: bool = False):
    """Classify, route, and respond to a query."""
    tier = classify_complexity(prompt)
    model_info = MODELS[tier]

    print(f"\n  [ SMART ROUTER ]")
    print(f"  - Query:      {prompt[:60]}{'...' if len(prompt)>60 else ''}")
    print(f"  - Complexity: {tier.upper()}")
    print(f"  - Model:      {model_info['id']} ({model_info['speed']})")
    print(f"  {'-' * 40}")

    # Check if the right model is loaded
    if not is_model_loaded(tier):
        current = get_loaded_model()
        log.info("Model mismatch: loaded=%s, need=%s", current, model_info["id"])
        log.info("Using currently loaded model: %s", current)
        log.info("For optimal routing, load with: lms load %s -y --gpu off", model_info["load_cmd"])

    if stream:
        stream_print(prompt)
    else:
        # Standard request
        payload = json.dumps({
            "model": get_loaded_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0,
        }).encode()

        t0 = time.time()
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        elapsed = time.time() - t0

        content = resp["choices"][0]["message"]["content"]
        comp = resp["usage"]["completion_tokens"]
        tok_s = comp / elapsed if elapsed > 0 else 0

        print(f"\n  {content[:500]}")
        print(f"\n  [{comp} tokens, {tok_s:.1f} tok/s, {elapsed:.1f}s]")


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Smart Model Router")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Auto-route a prompt")
    group.add_argument("--stream", type=str, help="Stream a response")
    group.add_argument("--classify", type=str, help="Classify complexity only")

    args = parser.parse_args()

    if args.classify:
        tier = classify_complexity(args.classify)
        model = MODELS[tier]
        print(f"  Complexity: {tier.upper()}")
        print(f"  Model:      {model['id']} ({model['speed']})")
    elif args.stream:
        route_and_respond(args.stream, stream=True)
    elif args.prompt:
        route_and_respond(args.prompt, stream=False)


if __name__ == "__main__":
    main()
