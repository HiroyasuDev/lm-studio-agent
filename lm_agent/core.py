import time
import asyncio
import logging
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
from lm_agent.config import config
from lm_agent.memory import prune_conversation, deduplicate_tool_calls
from lm_agent.tools import TOOLS, execute_tool

log = logging.getLogger("agent.core")

# Initializing with surgical precision for asynchronous IO
client = AsyncOpenAI(
    base_url=config.LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    timeout=config.DEFAULT_TIMEOUT,
)

_cached_model_id: Optional[str] = None

async def resolve_model_id() -> str:
    """Identify the loaded models asynchronously seamlessly."""
    global _cached_model_id
    if _cached_model_id:
        return _cached_model_id
    if config.MODEL_ID != "auto":
        _cached_model_id = config.MODEL_ID
        return config.MODEL_ID
    try:
        models = await client.models.list()
        if models.data:
            _cached_model_id = models.data[0].id
            return _cached_model_id
    except Exception:
        pass
    _cached_model_id = config.HF_TOKENIZER_PATH.split("/")[-1].lower() if config.HF_TOKENIZER_PATH else "default-model"
    return _cached_model_id

async def check_server() -> bool:
    """Determine asynchronously if the LM Studio API endpoint hosts a valid model."""
    try:
        models = await client.models.list()
        if not models.data:
            log.error("Server running but NO model loaded target bounds.")
            return False
        return True
    except Exception as e:
        log.error(f"Cannot reach server precisely: {e}")
        return False

async def chat_completion(
    messages: List[Dict[str, str]],
    *,
    temperature: float = config.DEFAULT_TEMPERATURE,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    json_mode: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute dynamic, non-blocking asynchronous chat completions adhering to rigorous backoff matrices."""
    model = await resolve_model_id()
    kwargs = {
        "model": model, 
        "messages": messages,
        "temperature": temperature, 
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {
            "type": "json_schema", 
            "json_schema": {"name": "response", "strict": True, "schema": {"type": "object", "additionalProperties": True}}
        }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    last_error = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            t0 = time.time()
            # Non-blocking invocation
            response = await client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0

            choice = response.choices[0]
            result = {
                "content": choice.message.content,
                "tool_calls": getattr(choice.message, "tool_calls", None),
                "finish_reason": choice.finish_reason,
            }
            if result["tool_calls"]:
                result["tool_calls"] = [
                    {"id": tc.id, "function_name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in result["tool_calls"]
                ]
            log.debug(f"Async LLM completion resolved sequentially in {elapsed:.2f}s")
            return result

        except Exception as e:
            last_error = e
            delay = config.RETRY_DELAY * (2 ** (attempt - 1))
            log.warning(f"Async attempt {attempt} failed: {e}. Non-blocking retry in {delay}s...")
            await asyncio.sleep(delay)

    raise RuntimeError(f"Uncompromising orchestration failed after {config.MAX_RETRIES} absolute retries.") from last_error

async def agentic_run(user_query: str) -> str:
    """Complete asynchronous sequential tool pipeline orchestration."""
    tool_call_history = []
    messages = [
        {"role": "system", "content": "You are a precise orchestrating logic gate. Do not deviate."},
        {"role": "user", "content": user_query},
    ]

    for round_num in range(1, 6):
        messages = prune_conversation(messages, config.MAX_CONTEXT_TOKENS)
        result = await chat_completion(messages, tools=TOOLS)

        if not result["tool_calls"]:
            return result["content"] or ""

        tool_calls = deduplicate_tool_calls(result["tool_calls"], tool_call_history)
        if not tool_calls:
            return result["content"] or ""

        messages.append({
            "role": "assistant", 
            "content": result["content"], 
            "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["function_name"], "arguments": tc["arguments"]}} for tc in tool_calls]
        })

        for tc in tool_calls:
            tool_result = execute_tool(tc["function_name"], tc["arguments"])
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

    messages.append({"role": "user", "content": "Conclusively summarize."})
    final_result = await chat_completion(messages)
    return final_result["content"] or ""
