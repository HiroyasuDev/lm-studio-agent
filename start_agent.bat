@echo off
REM ═══════════════════════════════════════════════════════════════
REM  LM Studio Agent – Quick Start
REM  Model  : Qwen2.5-1.5B-Instruct (Q4_0 GGUF)
REM  Server : http://<IP>:<PORT>
REM ═══════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║   LM Studio Agent Client                                ║
echo  ║   Qwen2.5-1.5B-Instruct · Dell                         ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM Check if LM Studio server is running
curl -s http://<IP>:<PORT>/v1/models >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  [!] LM Studio server is NOT running on port 1234.
    echo      Please start it:
    echo        1. Open LM Studio
    echo        2. Go to the "Developer" tab (left sidebar)
    echo        3. Load model: Qwen2.5-1.5B-Instruct
    echo        4. Click "Start Server"
    echo.
    pause
    exit /b 1
)

echo  [OK] LM Studio server detected.
echo.
echo  Choose a mode:
echo    1) Self-Test   – validate server + model
echo    2) Interactive – chat with the agent
echo    3) Exit
echo.
set /p choice="  Enter choice (1/2/3): "

if "%choice%"=="1" (
    py "%~dp0agent_client.py" --test
) else if "%choice%"=="2" (
    py "%~dp0agent_client.py" --interactive
) else (
    echo  Exiting.
)

echo.
pause
