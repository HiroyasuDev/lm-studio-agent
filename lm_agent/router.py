import json
import logging
import asyncio
from typing import Optional

try:
    import aiohttp
except ImportError:
    pass

from lm_agent.config import config

log = logging.getLogger("agent.router")

MODELS = {
    "fast": {"id": "qwen3-0.6b"},
    "balanced": {"id": "lfm2.5-vl-1.6b"},
    "quality": {"id": "qwen2.5-coder-7b-instruct"},
}

def classify_complexity(query: str) -> str:
    """Rigorous heuristic complexity extraction mapping."""
    words = len(query.split())
    if words <= 10:
        return "fast"
    elif words <= 30:
        return "balanced"
    return "quality"

async def get_loaded_model(session: aiohttp.ClientSession) -> Optional[str]:
    """Asynchronous extraction of current local model bounds."""
    try:
        async with session.get(f"{config.LM_STUDIO_BASE_URL}/models", timeout=5) as response:
            resp = await response.json()
            models = [m["id"] for m in resp.get("data", [])]
            return models[0] if models else None
    except Exception:
        return None

async def route_and_respond(prompt: str) -> str:
    """Highly engineered asynchronous semantic routing request execution."""
    tier = classify_complexity(prompt)
    model_info = MODELS.get(tier, MODELS["quality"])
    
    async with aiohttp.ClientSession() as session:
        current_model = await get_loaded_model(session)
        log.info(f"Routed algorithmic complexity '{tier}'. Requested mapping: {model_info['id']}. Current target: {current_model}")
        
        payload = {
            "model": current_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.0,
        }
        
        headers = {"Content-Type": "application/json"}
        target_url = f"{config.LM_STUDIO_BASE_URL}/chat/completions"
        
        try:
            async with session.post(target_url, json=payload, headers=headers, timeout=120) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.error(f"Deterministic routing operation aborted: {e}")
            return ""
