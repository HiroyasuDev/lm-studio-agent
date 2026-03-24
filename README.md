<div align="center">
  
# 🤖 LM Studio Agent (Production Stack) v2.1

A fully optimized, "10/10" local agentic AI stack designed for CPU inference. Featuring **live knowledge awareness**, **adaptive load-balancing**, and **smart complexity routing** to squeeze maximum performance out of DDR4 + CPU hardware constraints.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LM Studio](https://img.shields.io/badge/LM_Studio-0.4.2-purple.svg)](https://lmstudio.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🎯 Motivation & Purpose (5W1H)

| Dimension | Insight |
| :--- | :--- |
| 🧑‍💻 **Who** | Built for **solo developers, researchers**, and **engineers** working under strict local hardware constraints without sacrificing agentic AI capabilities. |
| 🛠️ **What** | A hyper-optimized, smart-routing **local AI stack** equipped with continuous real-time knowledge integration (RAG) and self-healing infrastructure. |
| 🖥️ **Where** | Designed specifically for **CPU-only inference environments** (e.g., standard enterprise workstations lacking discrete GPUs). |
| ⏱️ **When** | Created to immediately **bypass latency limiters** of running dense 7B models on standard DDR4 memory. |
| 💡 **Why** | Running local AI on CPU (especially complex agentic reasoning loops) is notoriously slow—often yielding single-digit tokens-per-second. The goal of this project isn't to buy better hardware, but to prove that **aggressive software optimization** (*live RAG tracking, dynamic thread pinning, context-aware routing*) can elevate a budget workstation to a "10/10" production-grade AI assistant. |
| ⚙️ **How** | Through a **synchronized multi-layer Python architecture** surrounding LM Studio: routing to hyper-fast 0.6B models for simple queries, reserving 7B models for dense code logic, dynamically throttling CPU load, and persisting zero-latency local knowledge via ChromaDB. |

---

## 🌟 The "10/10" Architecture

This stack transforms a standard LM Studio installation into a **reliable, self-healing, locally-aware AI agent**.

### 1. 🧠 Agentic Reliability (`agent_client.py`)
A modern orchestration agent built around the **Plan → Execute → Verify** loop:
* **Exponential Backoff:** Gracefully handles tool-call failures with `2s → 4s → 8s` retries.
* **JSON Output Validation:** Strict schema verification; automatically extracts JSON from model markdown.
* **Chain-of-Thought Verification:** The model self-evaluates critical answers with a confidence score.
* **True BPE Tokenization:** Exact programmatic token arrays via `tiktoken` (`cl100k_base`) guarantee zero context-window overflow during deep RAG retrieval.
* **Auto-Pruning Memory:** A sliding context window keeps exact tokens strictly mapped to the `4096` inference limit.
* **MCP Extensible:** Tested flawlessly with 9 MCP servers (Search, Docker, Filesystem, etc).

### 2. ⚡ Semantic Vector Routing (`smart_router.py`)
Hardware constraints demand extreme routing efficiency. The smart router vectorizes incoming prompts via `nomic-embed-text-v1.5` and mathematically computes **Cosine Similarity** against predefined domain centroids:
* **FAST Tier (Qwen3-0.6B) @ ~60 tok/s:** *Math, dates, single factoids.*
* **BALANCED Tier (LFM 1.6B) @ ~35 tok/s:** *Summarizations, lists, medium text.*
* **QUALITY Tier (Qwen2.5 7B Q6_K) @ ~12 tok/s:** *Complex coding (Python/SAS), step-by-step logic, detailed analysis.*
* **SSE Streaming:** Yields instant time-to-first-token. 

### 3. 👁️ Live Knowledge Awareness & Code-Aware Chunking (`rag_pipeline.py`)
You shouldn't have to tell your agent what you are working on.
* **Live Directory Watchdog:** Recursively watches your `Local` directory. When you save a file (`.py`, `.sas`, `.pdf`, `.docx`), it natively auto-ingests in the background.
* **Code-Aware AST Chunking:** Eliminates blind paragraph splitting. It maps native Abstract Syntax Trees (`ast.parse`) for Python (`def`/`class`), and enforces strict Procedural RegEx boundaries for SAS files, ensuring complex loops and `PROC SQL` macros are never structurally fragmented.
* **ChromaDB Vector Store:** Lightweight, persistent vector DB.
* **Nomic-Embed-Text-v1.5:** Local semantic embedding without API costs.

### 4. 🛡️ Production Watchdog (`service_wrapper.py`)
A self-healing infrastructure wrapper:
* **Adaptive Threading:** Throttles LM Studio CPU threads (from 8 to 4 P-Cores) when the underlying hardware is busy (e.g. running intensive SAS builds) to prevent OS stutter.
* **Heartbeat Monitoring:** Pings the model API every 30s.
* **Auto-Restart:** Automatically kills hung processes and hot-reloads the last known model if the server crashes.

---

## ⚙️ Hardware Optimization Profile

This stack is currently hyper-optimized for the **Intel Core i7-14700 (Desktop) + 64GB DDR4**:

* **P-Core Thread Pinning:** LM Studio `cpuThreads` clamped exclusively to 8 (ignoring E-Cores to prevent thread-thrashing latency).
* **High Batch Size:** `nBatch = 1024` for significantly faster prompt pre-processing.
* **Highest Precision Quantization:** Running **Q6_K** (6-bit weights) over the standard Q4 to preserve critical logical fidelity for dense coding tasks.

---

## 🚀 Quick Start

### Prerequisites
1. LM Studio running locally with the Server enabled (`0.0.0.0:1234`).
2. Required python dependencies: `pip install openai chromadb tiktoken numpy pypdf python-docx watchdog psutil`. 

### Commands 

**Run the Production Watchdog**
```powershell
# Monitors health, adapts CPU threading, and watches local files for auto-RAG 
py service_wrapper.py --watchdog
```

**Run a Single Smart Query**
```powershell
# Auto-routes to the best model based on complexity and streams the response
py smart_router.py --stream "What is the capital of Hawaii?"
```

**Run the Full Orchestrated Agent**
```powershell
# Invokes tools, builds a plan, executes, and verifies
py agent_client.py --agent "Can you retrieve the contents of my latest SAS build script?"
```

**Manual RAG Administration**
```powershell
py rag_pipeline.py status                # Check ChromaDB size and chunk count
py rag_pipeline.py search "dataset"      # Perform a raw semantic search
```

---

<div align="center">
  <i>"Local agentic AI is no longer a toy; it is a self-hosted colleague."</i>
</div>
