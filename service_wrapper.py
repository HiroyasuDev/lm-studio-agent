"""
Service Wrapper for LM Studio Agent
────────────────────────────────────
Health monitoring, auto-restart, watchdog process.

Usage:
  py service_wrapper.py              # start with health monitoring
  py service_wrapper.py --watchdog   # start with auto-restart watchdog
"""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from threading import Thread, Event

LOG_DIR = Path(r"D:\Local\Tools\LM_Studio\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"service_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("service")

LM_STUDIO_URL = "http://127.0.0.1:1234"
HEALTH_INTERVAL = 30       # seconds between health checks
RESTART_COOLDOWN = 60      # seconds between restart attempts
MAX_CONSECUTIVE_FAILS = 3  # failures before restart


# ═══════════════════════════════════════════════════════════════
#  Health Checks
# ═══════════════════════════════════════════════════════════════

def check_health() -> dict:
    """Check LM Studio server health. Returns status dict."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "server": "unknown",
        "model": None,
        "latency_ms": None,
    }
    try:
        t0 = time.time()
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/models",
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        latency = (time.time() - t0) * 1000

        models = [m["id"] for m in resp.get("data", [])]
        status["server"] = "healthy"
        status["model"] = models[0] if models else None
        status["latency_ms"] = round(latency, 1)

        if not models:
            status["server"] = "no_model"

    except urllib.error.URLError:
        status["server"] = "unreachable"
    except Exception as e:
        status["server"] = f"error: {e}"

    return status


def health_monitor(stop_event: Event):
    """Continuous health monitoring loop."""
    consecutive_fails = 0
    while not stop_event.is_set():
        status = check_health()

        if status["server"] == "healthy":
            consecutive_fails = 0
            log.info(
                "✓ Healthy | Model: %s | Latency: %sms",
                status["model"], status["latency_ms"]
            )
        else:
            consecutive_fails += 1
            log.warning(
                "✗ %s (fail %d/%d)",
                status["server"], consecutive_fails, MAX_CONSECUTIVE_FAILS
            )

        # Log to JSON file
        with open(LOG_DIR / "health.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(status) + "\n")

        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            log.error("Max failures reached. Attempting restart...")
            restart_lm_studio()
            consecutive_fails = 0
            time.sleep(RESTART_COOLDOWN)

        stop_event.wait(HEALTH_INTERVAL)


# ═══════════════════════════════════════════════════════════════
#  Auto-Restart
# ═══════════════════════════════════════════════════════════════

def find_lms_cli() -> str:
    """Find the lms CLI in PATH."""
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(p) / "lms.exe"
        if candidate.exists():
            return str(candidate)
        candidate = Path(p) / "lms"
        if candidate.exists():
            return str(candidate)
    # Common LM Studio locations
    for loc in [
        Path.home() / ".lmstudio" / "bin" / "lms.exe",
        Path(r"C:\Users\BinhPhan\AppData\Local\LM-Studio\resources\bin\lms.exe"),
    ]:
        if loc.exists():
            return str(loc)
    return "lms"


def restart_lm_studio():
    """Attempt to restart LM Studio server and reload model."""
    lms = find_lms_cli()
    log.info("Restarting LM Studio server via: %s", lms)
    try:
        # Start server
        subprocess.run(
            [lms, "server", "start"],
            capture_output=True, text=True, timeout=15
        )
        time.sleep(5)

        # Reload default model
        subprocess.run(
            [lms, "load", "qwen2.5-coder-7b-instruct", "-y",
             "--context-length", "4096", "--gpu", "off"],
            capture_output=True, text=True, timeout=60
        )
        log.info("Restart complete.")

    except subprocess.TimeoutExpired:
        log.error("Restart timed out.")
    except FileNotFoundError:
        log.error("lms CLI not found at: %s", lms)
    except Exception as e:
        log.error("Restart failed: %s", e)


# ═══════════════════════════════════════════════════════════════
#  Watchdog Mode
# ═══════════════════════════════════════════════════════════════

def watchdog():
    """Main watchdog: monitors health and auto-restarts on failure."""
    log.info("=" * 50)
    log.info("  SERVICE WATCHDOG STARTED")
    log.info("  Health check interval: %ds", HEALTH_INTERVAL)
    log.info("  Max consecutive fails: %d", MAX_CONSECUTIVE_FAILS)
    log.info("  Log dir: %s", LOG_DIR)
    log.info("=" * 50)

    # Initial health check
    status = check_health()
    if status["server"] != "healthy":
        log.warning("Server not healthy on start. Attempting restart...")
        restart_lm_studio()
        time.sleep(10)

    # Start monitoring
    stop = Event()
    try:
        health_monitor(stop)
    except KeyboardInterrupt:
        log.info("Watchdog stopped by user.")
        stop.set()


# ═══════════════════════════════════════════════════════════════
#  One-shot Health Check
# ═══════════════════════════════════════════════════════════════

def print_health():
    """Print a one-shot health status and exit."""
    status = check_health()
    print(f"\n  Server:  {status['server']}")
    print(f"  Model:   {status['model'] or 'none'}")
    print(f"  Latency: {status['latency_ms'] or 'N/A'}ms")
    print(f"  Time:    {status['timestamp']}")
    return 0 if status["server"] == "healthy" else 1


if __name__ == "__main__":
    if "--watchdog" in sys.argv:
        watchdog()
    else:
        sys.exit(print_health())
