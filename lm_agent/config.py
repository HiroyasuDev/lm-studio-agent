import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://<IP>:<PORT>/v1")
    MODEL_ID: str = os.getenv("MODEL_ID", "auto")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
    DEFAULT_TIMEOUT: int = int(os.getenv("DEFAULT_TIMEOUT", "120"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", "2"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "3500"))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "logs"))
    HF_TOKENIZER_PATH: str = os.getenv("HF_TOKENIZER_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")

config = Config()
config.LOG_DIR.mkdir(parents=True, exist_ok=True)
