# Werewolf Game

> **NOTE**: This project is still under active development. Please be aware of any bugs or crashes that it may cause while experimenting with it.

This is a Web-based game with LLM agents that simulates the social game called Werewolf or Mafia. It is built with LangChain, LangGraph, and Streamlit.

## How to run

1. After fetching the project, run the cli below to initialize the `.venv` and activate it.

```bash
uv venv

.venv/Scripts/activate
```

2. Use `uv add <package>` to install the necessary packages.

3. Create a new `.env` environment file using a environment template `.env.example` and fill the `AGENT_LLM_MODEL` value with the LLM model name, such as _llama3.2_.

4. Run `Streamlit` with the cli below.

```bash
streamlit run streamlit_app.py
```
