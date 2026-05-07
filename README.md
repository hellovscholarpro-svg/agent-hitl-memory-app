# Local Agentic RAG App with HITL and Memory

A Streamlit AI agent that runs without API keys. It reuses the local RAG architecture and Hugging Face open-source models, then adds:

- Human-in-the-loop plan approval
- Short-term memory from the current chat
- Long-term memory stored as local JSON
- RAG over uploaded documents
- No-key web search with `ddgs`
- Streamlit Cloud deployment readiness

## Default model choice

For easier Streamlit Cloud deployment, the default model is:

`Qwen/Qwen2.5-0.5B-Instruct`

The app also includes the prior DeepSeek model option:

`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

DeepSeek is better suited for local or hosted deployments with more memory/GPU.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to GitHub.
2. Create a Streamlit Community Cloud app from the repo.
3. Set `app.py` as the main file.
4. No secrets or API keys are required.
5. Keep the smaller default model for cloud deployment unless your app has enough memory.

## Optional environment variables

```bash
LOCAL_LLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
AGENT_MEMORY_DIR=.agent_memory
```

## Human-in-the-loop workflow

When HITL is enabled, the app first proposes a tool plan. The user can approve, remove actions, or deny the plan before any tools are run.

## Memory workflow

- Short-term memory: recent chat turns included in the prompt.
- Long-term memory: user-approved facts stored in `.agent_memory/long_term_memory.json`.
- Memory retrieval: relevant long-term memory items are embedded and retrieved for each question.

## Supported document types

- PDF
- DOCX
- TXT / MD
- CSV / JSON
- source-code text files
