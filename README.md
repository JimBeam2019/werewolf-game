# Werewolf Game

This project implements an AI-powered Werewolf social game in which multiple LLM agents simulate villagers and werewolves. The game supports multi-agent discussion, role-based reasoning, voting, night actions, and human participation through a Streamlit interface.

## Project Architecture

```text
streamlit_app.py
│
├── Presentation Layer
│   └── Streamlit user interface
│
├── Application Layer
│   ├── Game orchestration
│   ├── LangGraph workflows
│   └── Agent coordination
│
├── Domain Layer
│   ├── Game entities
│   ├── Game rules
│   ├── Player roles
│   └── Game state
│
└── Infrastructure Layer
    ├── LangChain LLM clients
    ├── vLLM integration
    ├── RAG retrieval
    ├── Vector store
    └── Long-term memory
```

## Environment Requirements

- Python 3.11 or later
- AMD Radeon GPU with ROCm support
- ROCm-compatible PyTorch
- vLLM with ROCm support
- Git

The application can also be developed using a CPU or another supported inference backend, although GPU inference is recommended for multi-agent interaction.

## How to run

### In a AMD Kadeon Cloud environment

| Tech Stack          | Name                      |
| ------------------- | ------------------------- |
| Inference Server    | vLLM                      |
| LLM Model           | Qwen/Qwen2.5-7B-Instruct  |
| LLM Model URL       | http://localhost:8000/v1  |
| Embedding Model     | Qwen/Qwen3-Embedding-0.6B |
| Embedding Model URL | http://localhost:8001/v1  |

1. After fetching the project, run the cli below to initialize the `.venv` and activate it.

```bash
uv venv

.venv/Scripts/activate
```

2. Use `uv sync` to install the necessary packages.

```bash
uv sync
```

3. Create a new `.env` environment file using a environment template `.env.example` and fill all the variables based on your choice. The below can be your reference.

```env
USE_VLLM=True
AGENT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_BASE_URL=http://localhost:8000/v1
EMBEDDING_LLM_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_LLM_BASE_URL=http://localhost:8001/v1
```

4. Start vLLM models in **separate terminal tabs**

- Embedding LLM Model

```bash
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8001 --runner pooling --gpu-memory-utilization 0.15
```

- LLM Model

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.75
```

5. Run `Streamlit` with the cli below.

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.baseUrlPath ""
```

6. Expose RC Tunnel

```bash
"$HOME/.local/bin/rc-tunnel" expose --port 8501
```

Open the URL displayed by the RC Tunnel, usually:

```text
https://rc-0123456789abcdef.radeon.firstdg.ai
```
