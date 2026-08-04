# Werewolf Game

> **NOTE**: This project is still under active development. Please be aware of any bugs or crashes that it may cause while experimenting with it.

This is a Web-based game with LLM agents that simulates the social game called Werewolf or Mafia. It is built with LangChain, LangGraph, and Streamlit.

## How to run

### In a local environment with `Ollama`

| Tech Stack       | Name                      |
| ---------------- | ------------------------- |
| Inference Server | Ollama                    |
| LLM Model        | llama3.2                  |
| LLM Model URL    | http://localhost:11434/v1 |
| Embedding Model  | qwen3-embedding           |

1. After fetching the project, run the cli below to initialize the `.venv` and activate it.

```bash
uv venv

.venv/Scripts/activate
```

2. Use `uv add <package>` to install the necessary packages.

3. Create a new `.env` environment file using a environment template `.env.example` and fill all the variables based on your choice. The below can be your reference.

```python
USE_VLLM=False
AGENT_LLM_MODEL=llama3.2:3b
LLM_BASE_URL=http://localhost:11434/v1
EMBEDDING_LLM_MODEL=qwen3-embedding
EMBEDDING_LLM_BASE_URL=http://localhost:8001/v1
```

4. Run `Streamlit` with the cli below.

```bash
streamlit run streamlit_app.py
```

### In a AMD Kadeon Cloud environment

| Tech Stack          | Name                       |
| ------------------- | -------------------------- |
| Inference Server    | vLLM                       |
| LLM Model           | Qwen/Qwen2.5-1.5B-Instruct |
| LLM Model URL       | http://localhost:8000/v1   |
| Embedding Model     | Qwen/Qwen3-Embedding-0.6B  |
| Embedding Model URL | http://localhost:8001/v1   |

1. After fetching the project, run the cli below to initialize the `.venv` and activate it.

```bash
uv venv

.venv/Scripts/activate
```

2. Use `uv add <package>` to install the necessary packages.

3. Create a new `.env` environment file using a environment template `.env.example` and fill all the variables based on your choice. The below can be your reference.

```python
USE_VLLM=True
AGENT_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LLM_BASE_URL=http://localhost:8000/v1
EMBEDDING_LLM_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_LLM_BASE_URL=http://localhost:8001/v1
```

4. Start vLLM models

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000 --gpu-memory-utilization 0.75

vllm serve Qwen/Qwen3-Embedding-0.6B --port 8001 --runner pooling --gpu-memory-utilization 0.15
```

5. Run `Streamlit` with the cli below.

```bash
streamlit run test.py --server.address 0.0.0.0 --server.port 8501 --server.baseUrlPath ""
```

6. Expose RC Tunnel

```bash
"$HOME/.local/bin/rc-tunnel" expose --port 8501
```
