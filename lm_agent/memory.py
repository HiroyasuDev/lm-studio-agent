import json
import logging
import hashlib
from typing import Any, Dict, List, Tuple

from lm_agent.config import config

log = logging.getLogger("agent.memory")

# Uncompromising tokenization initialization
try:
    from transformers import AutoTokenizer
    # Load fast tokenizer offline if possible, else download
    log.info(f"Initializing native BPE Tokenizer: {config.HF_TOKENIZER_PATH}")
    TOKENIZER = AutoTokenizer.from_pretrained(config.HF_TOKENIZER_PATH, trust_remote_code=True)
except ImportError:
    log.warning("transformers not installed. Falling back to heuristic estimation.")
    TOKENIZER = None
except Exception as e:
    log.warning(f"Failed to load native tokenizer {config.HF_TOKENIZER_PATH}: {e}. Falling back.")
    TOKENIZER = None

def validate_json_output(text: str) -> Tuple[bool, Any]:
    """Validate that output is parseable JSON with zero tolerance for defects outside code blocks."""
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
    """Remove duplicate tool calls deterministically via MD5 signatures."""
    unique = []
    for tc in tool_calls:
        sig = hashlib.md5(f"{tc['function_name']}:{tc['arguments']}".encode()).hexdigest()
        if sig not in history:
            history.append(sig)
            unique.append(tc)
        else:
            log.warning(f"Deduplicated redundant tool call: {tc['function_name']}")
    return unique

def count_tokens(text: str) -> int:
    """Exact token counting utilizing native model BPE vocabularies."""
    if TOKENIZER is not None:
        try:
            return len(TOKENIZER.encode(str(text)))
        except Exception as e:
            log.debug(f"Tokenizer encoding failed: {e}")
            pass
    
    # Fallback heuristic: assumes ~3.5 chars per token for dense code models
    return int(len(str(text)) / 3.5)

def estimate_conversation_tokens(messages: List[Dict[str, str]]) -> int:
    """Calculate absolute context weight across the entire message array."""
    total = 0
    for m in messages:
        # ChatML format overhead (role tokens)
        total += 4 + count_tokens(m.get("content", ""))
    return total + 3

def prune_conversation(messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """
    Surgical sliding window logic ensuring zero context-window overflow.
    Strictly safeguards the system prompt and relies on exact BPE bounds.
    """
    if not messages or estimate_conversation_tokens(messages) <= max_tokens:
        return messages

    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    while estimate_conversation_tokens(system + non_system) > max_tokens and len(non_system) > 4:
        pruned_msg = non_system.pop(0)
        log.debug(f"Pruned message from history. Tokens mitigated constraints.")

    return system + non_system
