# Local Agent Workspace Setup

This directory contains starter templates to get your local "agentic" architecture running entirely via **LM Studio** and your **CPU + 64GB RAM** setup!

## 🚀 1. Setup Your Virtual Environment
Open a terminal in PowerShell and run:

```powershell
# Navigate to this folder
cd D:\Local\agent-workspace

# Create a virtual environment using the standard Windows python launcher
py -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install the agent frameworks
.\venv\Scripts\pip.exe install -r requirements.txt
```

## 🧠 2. Spin up your Local Model in LM Studio
1. Open **LM Studio**.
2. Download a model that fits well within your RAM (e.g., **Mistral Nemo 12B Instruct** or **Nemotron-3-Nano**).
3. Go to the **Local Server** (↔) tab on the left.
4. Ensure the port is set to `1234` (the default).
5. Click **Start Server**. 

*Behind the scenes, LM Studio acts exactly like the remote OpenAI API, so these frameworks don't realize they're actually "talking" to your local hardware!*

## 🤖 3. Run Your Framework of Choice

### Option A: The Deterministic Pipeline (PydanticAI)
**Best for**: Highly structured data extraction, software routing, or strict tool completion. 
```powershell
python pydantic_agent.py
```
This script forces the Mistral Nemo model to return a heavily strictly typed Python object containing `summary`, `key_points`, and a `confidence_score`.

### Option B: The Multi-Agent Organization (CrewAI)
**Best for**: "Chain of Thought" research, iterative writing, and persona-driven AI workflows.
```powershell
python crewai_agent.py
```
This scripts spins up a "Researcher Manager" and a "Tech Blogger". The model effectively talks to itself (first acting as the researcher, then passing the baton to the writer) without ever reaching out to the cloud.
