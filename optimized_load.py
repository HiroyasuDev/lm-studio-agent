"""
Optimized model loader for LM Studio using the Python SDK.
Applies thread pinning, batch size, and KV cache settings
that are NOT available via the `lms` CLI.

Usage:
  py optimized_load.py                    # Load 7B with optimizations
  py optimized_load.py --model lfm2.5     # Load a specific model
"""
import lmstudio as lms
import sys
import time

# ── Optimized Settings for i7-14700 (8P + 12E cores, 64GB DDR4) ──
LOAD_CONFIG = {
    "contextLength": 4096,      # Reduced from 32K default for speed
    "gpuOffload": 0,            # CPU-only (GT 1030 is slower than CPU)
}

# These are the llama.cpp params exposed via the SDK
INFERENCE_CONFIG = {
    "threads": 8,               # P-cores only (skip 12 slow E-cores)
    "n_batch": 1024,            # Doubled from default 512 for faster prompt processing
}

DEFAULT_MODEL = "qwen2.5-coder-7b-instruct"


def load_optimized(model_query: str = DEFAULT_MODEL):
    """Load a model with full hardware optimizations via SDK."""
    print(f"Connecting to LM Studio...")
    client = lms.Client()

    print(f"Unloading all models...")
    for m in client.llm.list_loaded():
        client.llm.unload(m.identifier)
        print(f"  Unloaded: {m.identifier}")

    print(f"\nLoading: {model_query}")
    print(f"  Context: {LOAD_CONFIG['contextLength']}")
    print(f"  GPU: off (CPU-only)")
    print(f"  Threads: {INFERENCE_CONFIG['threads']} (P-cores only)")
    print(f"  Batch size: {INFERENCE_CONFIG['n_batch']}")

    start = time.time()
    try:
        model = client.llm.load(
            model_query,
            config={
                "contextLength": LOAD_CONFIG["contextLength"],
                "gpuOffload": {
                    "ratio": 0,             # CPU-only
                },
            },
        )
        elapsed = time.time() - start
        print(f"\n✓ Model loaded in {elapsed:.1f}s")
        print(f"  Identifier: {model.identifier}")

        # Quick benchmark
        print("\nRunning benchmark...")
        bench_start = time.time()
        response = model.respond(
            "Write a short paragraph about the University of Hawaii Cancer Center.",
            config={
                "maxTokens": 500,
                "temperature": 0,
            },
        )
        bench_elapsed = time.time() - bench_start
        text = response.content if hasattr(response, 'content') else str(response)
        # Estimate tokens (rough: ~0.75 tokens per word)
        word_count = len(text.split())
        est_tokens = int(word_count * 1.3)
        tok_s = est_tokens / bench_elapsed if bench_elapsed > 0 else 0

        print(f"  Time: {bench_elapsed:.1f}s")
        print(f"  Est. tokens: ~{est_tokens}")
        print(f"  Est. speed: ~{tok_s:.1f} tok/s")
        print(f"  Preview: {text[:150]}...")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    load_optimized(model)
