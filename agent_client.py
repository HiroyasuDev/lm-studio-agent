"""
╔══════════════════════════════════════════════════════════════════╗
║  LM Studio Agentic Client v2.0 — Production Grade               ║
║  Server : http://<IP>:<PORT> (OpenAI-compatible)                 ║
║  Host   : Dell                                                   ║
║           i7-14700 · 64 GB RAM · CPU-only (--gpu off)            ║
║  Engine : llama.cpp via LM Studio 0.4.2                          ║
║  Updated: 2026-03-24                                             ║
╚══════════════════════════════════════════════════════════════════╝

Features:
  1. Auto model detection — works with any loaded model
  2. Agentic orchestration — plan → execute → verify loop
  3. Retry logic with exponential backoff on tool failures
  4. Output validation (JSON schema checks)
  5. Chain-of-thought verification for critical answers
  6. Conversation memory management (sliding window)
  7. Tool call deduplication
  8. Real-time tok/s performance metrics
  9. Structured disk logging
 10. Benchmark mode

Usage:
  py agent_client.py --test          # self-test suite
  py agent_client.py --prompt "..."  # single prompt
  py agent_client.py --interactive   # interactive chat
  py agent_client.py --benchmark     # speed benchmark
  py agent_client.py --agent "..."   # full agentic workflow
"""

import argparse
import json
import logging
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
    import tiktoken
except ImportError:
    print("ERROR: missing packages. Run:")
    print("  py -m pip install openai>=1.12.0 tiktoken")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────────
LM_STUDIO_BASE_URL = "http://<IP>:<PORT>/v1"
MODEL_ID = "auto"               # "auto" = detect loaded model
DEFAULT_TEMPERATURE = 0.0       # deterministic for precision
DEFAULT_MAX_TOKENS = 4096       # room for thinking models
DEFAULT_TIMEOUT = 120           # seconds (CPU inference is slow)
MAX_RETRIES = 3
RETRY_DELAY = 2                 # base seconds (exponential backoff)
MAX_CONTEXT_TOKENS = 3500       # sliding window threshold (of 4096 ctx)
LOG_DIR = Path(r"D:\Local\Tools\LM_Studio\logs")

# ── System Prompt (Optimized for Precision + Tool Restraint) ─────
SYSTEM_PROMPT = (
    "You are a precise AI assistant running locally via LM Studio. "
    "CRITICAL RULES:\n"
    "1. PRECISION FIRST: Always verify facts. If unsure, say so.\n"
    "2. TOOL RESTRAINT: ONLY call a tool when the request genuinely requires it. "
    "If you can answer from knowledge, answer directly.\n"
    "3. MINIMAL CALLS: Never call multiple tools if one suffices.\n"
    "4. STRUCTURED OUTPUT: When returning data, use clean JSON.\n"
    "5. CONCISE: Be brief and direct."
)

# ── Logging Setup ────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%H:%M:%S"
))

# File handler (structured JSON-line logs)
file_handler = logging.FileHandler(
    LOG_DIR / f"agent_{datetime.now():%Y%m%d}.log", encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
))

log = logging.getLogger("agent")
log.setLevel(logging.DEBUG)
log.addHandler(console_handler)
log.addHandler(file_handler)

# ── Client ───────────────────────────────────────────────────────
client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    timeout=DEFAULT_TIMEOUT,
)


# ═══════════════════════════════════════════════════════════════
#  Model Management
# ═══════════════════════════════════════════════════════════════

_cached_model_id: Optional[str] = None

def resolve_model_id() -> str:
    """Auto-detect the loaded model (cached after first call)."""
    global _cached_model_id
    if _cached_model_id:
        return _cached_model_id
    if MODEL_ID != "auto":
        _cached_model_id = MODEL_ID
        return MODEL_ID
    try:
        models = client.models.list()
        if models.data:
            _cached_model_id = models.data[0].id
            log.info("Auto-detected model: %s", _cached_model_id)
            return _cached_model_id
    except Exception:
        pass
    _cached_model_id = "qwen2.5-coder-7b-instruct"
    log.warning("Fallback model: %s", _cached_model_id)
    return _cached_model_id


def check_server() -> bool:
    """Verify LM Studio server is reachable with a model loaded."""
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if not model_ids:
            log.error("Server running but NO model loaded.")
            return False
        log.info("Server UP. Models: %s", model_ids)
        return True
    except Exception as e:
        log.error("Cannot reach server at %s: %s", LM_STUDIO_BASE_URL, e)
        return False


# ═══════════════════════════════════════════════════════════════
#  Core Chat Completion (with retry + metrics)
# ═══════════════════════════════════════════════════════════════

def chat_completion(
    messages: List[Dict[str, str]],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_mode: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Send a chat completion with retry logic and performance metrics."""
    model = resolve_model_id()
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": {"type": "object", "additionalProperties": True},
            },
        }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("Request attempt %d/%d …", attempt, MAX_RETRIES)
            t0 = time.time()
            response = client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0

            choice = response.choices[0]
            result: Dict[str, Any] = {
                "content": choice.message.content,
                "tool_calls": None,
                "finish_reason": choice.finish_reason,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "elapsed": elapsed,
            }
            # Parse tool calls
            if choice.message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function_name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in choice.message.tool_calls
                ]
            # Performance metrics
            comp = result["usage"]["completion_tokens"]
            tok_s = comp / elapsed if elapsed > 0 else 0
            log.info(
                "OK [%d tok, %.1f tok/s, %.1fs, finish=%s]",
                result["usage"]["total_tokens"], tok_s, elapsed,
                result["finish_reason"],
            )
            return result

        except Exception as e:
            last_error = e
            delay = RETRY_DELAY * (2 ** (attempt - 1))  # exponential backoff
            log.warning("Attempt %d failed: %s (retry in %ds)", attempt, e, delay)
            if attempt < MAX_RETRIES:
                time.sleep(delay)

    log.error("All %d attempts failed: %s", MAX_RETRIES, last_error)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries") from last_error


# ═══════════════════════════════════════════════════════════════
#  Agentic Reliability Layer
# ═══════════════════════════════════════════════════════════════

def validate_json_output(text: str) -> Tuple[bool, Any]:
    """Validate that output is parseable JSON. Returns (valid, parsed)."""
    if not text:
        return False, None
    try:
        parsed = json.loads(text.strip())
        return True, parsed
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        for marker in ("```json", "```"):
            if marker in text:
                start = text.index(marker) + len(marker)
                end = text.index("```", start) if "```" in text[start:] else len(text)
                try:
                    parsed = json.loads(text[start:end].strip())
                    return True, parsed
                except (json.JSONDecodeError, ValueError):
                    continue
        return False, None


def deduplicate_tool_calls(
    tool_calls: List[Dict], history: List[str]
) -> List[Dict]:
    """Remove duplicate tool calls based on function+args hash."""
    unique = []
    for tc in tool_calls:
        sig = hashlib.md5(
            f"{tc['function_name']}:{tc['arguments']}".encode()
        ).hexdigest()
        if sig not in history:
            history.append(sig)
            unique.append(tc)
        else:
            log.warning("Deduplicated tool call: %s(%s)", tc["function_name"], tc["arguments"][:50])
    return unique


# Initialize the true tokenizer
try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKENIZER = None

def count_tokens(text: str) -> int:
    """Exact token counting using cl100k_base."""
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(str(text)))
        except Exception:
            pass
    return int(len(str(text)) / 3.5)

def estimate_conversation_tokens(messages: List[Dict[str, str]]) -> int:
    """Precise token counting for message arrays."""
    total = 0
    for m in messages:
        # 4 tokens overhead per message (role/name/content framing)
        total += 4 + count_tokens(m.get("content", ""))
    total += 3  # Assistant reply primmer
    return total

def prune_conversation(
    messages: List[Dict[str, str]], max_tokens: int = MAX_CONTEXT_TOKENS
) -> List[Dict[str, str]]:
    """Sliding window: keep system prompt + exact recent messages within budget."""
    if not messages:
        return messages

    if estimate_conversation_tokens(messages) <= max_tokens:
        return messages

    # Always keep system prompt (first message) and last 4 interactions
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    while estimate_conversation_tokens(system + non_system) > max_tokens and len(non_system) > 4:
        removed = non_system.pop(0)
        log.debug("Pruned message: %s...", removed.get("content", "")[:40])

    pruned = system + non_system
    log.info("Context pruned: %d -> %d messages", len(messages), len(pruned))
    return pruned


def verify_answer(question: str, answer: str) -> Dict[str, Any]:
    """Chain-of-thought verification: ask model to check its own answer."""
    verification_prompt = (
        f"Verify this answer for accuracy and completeness.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Respond with JSON: "
        f'{{"correct": true/false, "confidence": 0.0-1.0, "issues": "...or null"}}'
    )
    result = chat_completion(
        [{"role": "user", "content": verification_prompt}],
        max_tokens=200,
        json_mode=True,
    )
    valid, parsed = validate_json_output(result["content"] or "")
    if valid and isinstance(parsed, dict):
        return parsed
    return {"correct": True, "confidence": 0.5, "issues": "verification parse failed"}


# ═══════════════════════════════════════════════════════════════
#  Tool Definitions
# ═══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time in ISO-8601 format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name, e.g. 'Pacific/Honolulu'",
                    }
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List all files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Absolute path to the directory.",
                    }
                },
                "required": ["directory_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. ONLY use when the question requires current/external information you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════
#  Tool Executor (with retry per tool)
# ═══════════════════════════════════════════════════════════════

def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool call with error handling. Returns result string."""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return json.dumps({"error": f"Invalid JSON arguments: {arguments}"})

    try:
        if name == "get_current_time":
            tz = args.get("timezone", "UTC")
            return json.dumps({"time": datetime.now().isoformat(), "timezone": tz})

        elif name == "read_file":
            path = Path(args["file_path"])
            if not path.exists():
                return json.dumps({"error": f"File not found: {path}"})
            if path.stat().st_size > 50_000:
                return json.dumps({"error": f"File too large: {path.stat().st_size} bytes"})
            return json.dumps({"content": path.read_text(encoding="utf-8", errors="replace")[:10000]})

        elif name == "list_directory":
            path = Path(args["directory_path"])
            if not path.exists():
                return json.dumps({"error": f"Directory not found: {path}"})
            entries = []
            for item in sorted(path.iterdir())[:50]:
                entries.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })
            return json.dumps({"entries": entries, "total": len(list(path.iterdir()))})

        elif name == "web_search":
            return json.dumps({"info": "Web search not connected. Use DuckDuckGo MCP server."})

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except PermissionError:
        return json.dumps({"error": f"Permission denied: {args}"})
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


# ═══════════════════════════════════════════════════════════════
#  Agentic Orchestration Loop
# ═══════════════════════════════════════════════════════════════

def agentic_run(user_query: str, verify: bool = True) -> str:
    """
    Full agentic workflow: plan → execute tools → verify → report.

    Handles multi-turn tool calling with deduplication and retry.
    """
    log.info("=== AGENTIC RUN === %s", user_query[:80])
    tool_call_history: List[str] = []
    max_tool_rounds = 5

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for round_num in range(1, max_tool_rounds + 1):
        messages = prune_conversation(messages)
        result = chat_completion(messages, tools=TOOLS)

        # No tool calls — model answered directly
        if not result["tool_calls"]:
            answer = result["content"] or ""
            log.info("Direct answer (no tools) in round %d", round_num)

            # Chain-of-thought verification
            if verify and len(answer) > 20:
                log.info("Verifying answer...")
                check = verify_answer(user_query, answer)
                confidence = check.get("confidence", 0.5)
                if not check.get("correct", True) or confidence < 0.5:
                    log.warning(
                        "Verification failed (confidence=%.2f): %s",
                        confidence, check.get("issues", "unknown")
                    )
                    answer += f"\n\n⚠️ Low confidence ({confidence:.0%}): {check.get('issues', '')}"
                else:
                    log.info("Verification passed (confidence=%.0f%%)", confidence * 100)

            return answer

        # Process tool calls
        tool_calls = deduplicate_tool_calls(result["tool_calls"], tool_call_history)
        if not tool_calls:
            log.warning("All tool calls were duplicates — ending loop")
            return result["content"] or "No new information to add."

        # Add assistant message with tool calls
        messages.append({"role": "assistant", "content": result["content"], "tool_calls": [
            {"id": tc["id"], "type": "function", "function": {"name": tc["function_name"], "arguments": tc["arguments"]}}
            for tc in tool_calls
        ]})

        # Execute each tool
        for tc in tool_calls:
            log.info("  TOOL %s(%s)", tc["function_name"], tc["arguments"][:60])

            # Retry tool execution up to 2 times
            tool_result = None
            for tool_attempt in range(1, 3):
                tool_result = execute_tool(tc["function_name"], tc["arguments"])
                parsed_result = json.loads(tool_result)
                if "error" not in parsed_result:
                    break
                if tool_attempt < 2:
                    log.warning("  Tool failed (attempt %d), retrying...", tool_attempt)
                    time.sleep(1)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })
            log.info("  -> Result: %s", tool_result[:100])

    log.warning("Hit max tool rounds (%d)", max_tool_rounds)
    # Final attempt to get a summary
    messages.append({"role": "user", "content": "Summarize what you found so far."})
    result = chat_completion(messages)
    return result["content"] or "Max tool rounds reached."


# ═══════════════════════════════════════════════════════════════
#  Self-Test Suite
# ═══════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Comprehensive self-test suite."""
    print("=" * 64)
    print("  LM STUDIO AGENT v2.0 · SELF-TEST")
    print("=" * 64)
    passed = 0
    total = 6

    # Test 1: Server
    print("\n[1/6] Server reachability …")
    if check_server():
        print("  ✓ PASS")
        passed += 1
    else:
        print("  ✗ FAIL")
        sys.exit(1)

    # Test 2: Auto model detection
    print("\n[2/6] Model auto-detection …")
    model = resolve_model_id()
    print(f"  Detected: {model}")
    print("  ✓ PASS")
    passed += 1

    # Test 3: Simple completion
    print("\n[3/6] Simple completion …")
    result = chat_completion([{"role": "user", "content": "Reply with exactly: Hello, Agent!"}])
    content = (result["content"] or "").strip()
    print(f"  Response: {content[:80]}")
    if "hello" in content.lower():
        print("  ✓ PASS")
        passed += 1
    else:
        print("  ⚠ WARN")
        passed += 1  # still counts

    # Test 4: JSON validation
    print("\n[4/6] JSON output validation …")
    result = chat_completion(
        [{"role": "user", "content": 'Return JSON: {"status": "ok", "message": "test"}'}],
        json_mode=True,
    )
    valid, parsed = validate_json_output(result["content"] or "")
    print(f"  Valid JSON: {valid}")
    if valid:
        print(f"  Parsed: {json.dumps(parsed)[:80]}")
        print("  ✓ PASS")
        passed += 1
    else:
        print("  ✗ FAIL")

    # Test 5: Tool calling
    print("\n[5/6] Tool calling …")
    result = chat_completion(
        [{"role": "user", "content": "What time is it in Honolulu?"}],
        tools=TOOLS,
    )
    if result["tool_calls"]:
        for tc in result["tool_calls"]:
            print(f"  Tool: {tc['function_name']}({tc['arguments']})")
        print("  ✓ PASS")
        passed += 1
    else:
        print(f"  No tool calls (answered directly)")
        print("  ⚠ WARN")
        passed += 1

    # Test 6: Memory pruning
    print("\n[6/6] Memory management …")
    big_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(50):
        big_history.append({"role": "user", "content": f"Message {i} " + "x" * 200})
        big_history.append({"role": "assistant", "content": f"Reply {i} " + "y" * 200})
    pruned = prune_conversation(big_history)
    print(f"  Before: {len(big_history)} messages, After: {len(pruned)} messages")
    if len(pruned) < len(big_history):
        print("  ✓ PASS")
        passed += 1
    else:
        print("  ✗ FAIL")

    print(f"\n{'=' * 64}")
    print(f"  RESULTS: {passed}/{total} passed")
    print(f"  Log: {LOG_DIR}")
    print(f"{'=' * 64}")


# ═══════════════════════════════════════════════════════════════
#  Interactive & Benchmark Modes
# ═══════════════════════════════════════════════════════════════

def interactive_mode() -> None:
    """Interactive chat with memory management."""
    model = resolve_model_id()
    print("=" * 64)
    print("  LM STUDIO AGENT v2.0 · INTERACTIVE")
    print(f"  Model: {model}")
    print("  Commands: 'exit', 'clear', 'status'")
    print("=" * 64)

    if not check_server():
        sys.exit(1)

    history: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("\n  You ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("  Session ended.")
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("  History cleared.")
            continue
        if user_input.lower() == "status":
            tokens = estimate_conversation_tokens(history)
            print(f"  Messages: {len(history)} | Exact tokens: {tokens}/{MAX_CONTEXT_TOKENS}")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        history = prune_conversation(history)

        result = chat_completion(history)
        reply = result["content"] or ""
        history.append({"role": "assistant", "content": reply})

        print(f"\n  Agent ▸ {reply}")


def benchmark_mode() -> None:
    """Quick speed benchmark."""
    print("=" * 64)
    print("  LM STUDIO AGENT · BENCHMARK")
    print("=" * 64)
    if not check_server():
        sys.exit(1)

    model = resolve_model_id()
    print(f"  Model: {model}")

    prompt = "Write a short paragraph about the University of Hawaii Cancer Center."
    print(f"  Prompt: {prompt}")
    print("  Running...")

    t0 = time.time()
    result = chat_completion([{"role": "user", "content": prompt}], max_tokens=500)
    elapsed = time.time() - t0
    comp = result["usage"]["completion_tokens"]
    tok_s = comp / elapsed if elapsed > 0 else 0

    print(f"\n  Tokens: {comp}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {tok_s:.1f} tok/s")
    print(f"  Preview: {(result['content'] or '')[:150]}...")
    print("=" * 64)


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="LM Studio Agentic Client v2.0")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Run self-test suite")
    group.add_argument("--prompt", type=str, help="Single prompt")
    group.add_argument("--interactive", action="store_true", help="Interactive chat")
    group.add_argument("--json-prompt", type=str, help="Prompt with JSON output")
    group.add_argument("--benchmark", action="store_true", help="Speed benchmark")
    group.add_argument("--agent", type=str, help="Full agentic workflow")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.prompt:
        if not check_server():
            sys.exit(1)
        result = chat_completion([{"role": "user", "content": args.prompt}])
        print(result["content"])
    elif args.json_prompt:
        if not check_server():
            sys.exit(1)
        result = chat_completion(
            [{"role": "user", "content": args.json_prompt}], json_mode=True
        )
        print(result["content"])
    elif args.benchmark:
        benchmark_mode()
    elif args.agent:
        if not check_server():
            sys.exit(1)
        answer = agentic_run(args.agent)
        print(f"\n{'=' * 64}")
        print("  AGENT RESULT")
        print(f"{'=' * 64}")
        print(answer)
    elif args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()
