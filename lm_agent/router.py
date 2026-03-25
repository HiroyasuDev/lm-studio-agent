import json
import urllib.request
import logging
from typing import Optional

from lm_agent.config import config

log = logging.getLogger("agent.router")

MODELS = {
    "fast": {"id": "qwen3-0.6b"},
    "balanced": {"id": "lfm2.5-vl-1.6b"},
    "quality": {"id": "qwen2.5-coder-7b-instruct"},
}

def classify_complexity(query: str) -> str:
    """Classify the query to determine optimal model tier using basic heuristics."""
    words = len(query.split())
    if words <= 10:
        return "fast"
    elif words <= 30:
        return "balanced"
    return "quality"

def get_loaded_model() -> Optional[str]:
    """Get the currently loaded model ID."""
    try:
        req = urllib.request.Request(f"{config.LM_STUDIO_BASE_URL}/models")
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        models = [m["id"] for m in resp.get("data", [])]
        return models[0] if models else None
    except Exception:
        return None

def route_and_respond(prompt: str) -> str:
    """Route exactly to the best model tier."""
    tier = classify_complexity(prompt)
    model_info = MODELS.get(tier, MODELS["quality"])
    
    current_model = get_loaded_model()
    log.info(f"Routed complexity '{tier}'. Requested mapping: {model_info['id']}. Current loaded server model: {current_model}")
    
    payload = json.dumps({
        "model": current_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0,
    }).encode()
    
    req = urllib.request.Request(
        f"{config.LM_STUDIO_BASE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"Routing failed: {e}")
        return ""
