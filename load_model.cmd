@echo off
REM ══════════════════════════════════════════════════════════════
REM  Optimized Model Loader for <MACHINE_MODEL>
REM  <MACHINE_SPECS>
REM  Priority: Precision > Speed > Tool Calling
REM ══════════════════════════════════════════════════════════════
REM  Usage:
REM    load_model.cmd                     (loads 7B quality model)
REM    load_model.cmd fast                (loads LFM 2.5 speed model)
REM    load_model.cmd gemma               (loads Gemma 3n E4B)
REM    load_model.cmd qwen3               (loads Qwen3 0.6B thinking)
REM ══════════════════════════════════════════════════════════════

set MODE=%1
if "%MODE%"=="" set MODE=quality

echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║  LM Studio Optimized Loader               ║
echo  ║  Mode: %MODE%                              
echo  ╚═══════════════════════════════════════════╝
echo.

REM Start server if not running
lms server start 2>nul

REM Unload any existing model
lms unload --all 2>nul

if "%MODE%"=="quality" (
    echo  Loading: Qwen2.5-Coder-7B [Precision Mode]
    echo  Speed: ~9.3 tok/s │ RAM: 4.68 GB │ Best for: complex coding
    lms load qwen2.5-coder-7b-instruct -y --context-length 4096 --gpu off
)

if "%MODE%"=="fast" (
    echo  Loading: LFM 2.5-VL-1.6B [Speed Mode]
    echo  Speed: ~35 tok/s │ RAM: 1.58 GB │ Best for: quick queries
    lms load lmstudio-community/LFM2.5-VL-1.6B-GGUF -y --context-length 4096 --gpu off
)

if "%MODE%"=="gemma" (
    echo  Loading: Gemma 3n E4B [Balanced Mode]
    echo  Speed: ~14 tok/s │ RAM: 4.24 GB │ Best for: general tasks
    lms load gemma-3n-e4b-it -y --context-length 4096 --gpu off
)

if "%MODE%"=="qwen3" (
    echo  Loading: Qwen3 0.6B [Thinking Mode]
    echo  Speed: ~64 tok/s │ RAM: 805 MB │ Best for: fast reasoning
    lms load lmstudio-community/Qwen3-0.6B-GGUF -y --context-length 4096 --gpu off
)

echo.
echo  ⚡ IMPORTANT: Open LM Studio GUI and set:
echo     1. Threads = 8  (P-cores only, skip slow E-cores)
echo     2. Batch Size = 1024
echo     3. KV Cache Type = Q8_0
echo.
echo  Ready. Run benchmark: py agent_client.py --benchmark
