import json
import logging
import hashlib
from typing import Any, Dict, List, Tuple

try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TOKENIZER = None

log = logging.getLogger("agent.memory")

def validate_json_output(text: str) -> Tuple[bool, Any]:
    """Validate that output is parseable JSON."""
    if not text:
        return False, None
    try:
        parsed = json.loads(text.strip())
        return True, parsed
    except json.JSONDecodeError:
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

def deduplicate_tool_calls(tool_calls: List[Dict], history: List[str]) -> List[Dict]:
    """Remove duplicate tool calls based on signature."""
    unique = []
    for tc in tool_calls:
        sig = hashlib.md5(f"{tc['function_name']}:{tc['arguments']}".encode()).hexdigest()
        if sig not in history:
            history.append(sig)
            unique.append(tc)
        else:
            log.warning("Deduplicated tool call: %s", tc["function_name"])
    return unique

def count_tokens(text: str) -> int:
    """Exact token counting using cl100k_base."""
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(str(text)))
        except Exception:
            pass
    return int(len(str(text)) / 3.5)

def estimate_conversation_tokens(messages: List[Dict[str, str]]) -> int:
    total = 0
    for m in messages:
        total += 4 + count_tokens(m.get("content", ""))
    return total + 3

def prune_conversation(messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """Sliding window to keep conversation within token limits."""
    if not messages or estimate_conversation_tokens(messages) <= max_tokens:
        return messages

    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    while estimate_conversation_tokens(system + non_system) > max_tokens and len(non_system) > 4:
        non_system.pop(0)

    return system + non_system
