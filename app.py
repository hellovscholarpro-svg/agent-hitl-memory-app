import io
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup
from ddgs import DDGS
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

APP_TITLE = "Local Agentic RAG: HITL + Memory"
DEFAULT_LLM = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEEPSEEK_LLM = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MEMORY_DIR = Path(os.getenv("AGENT_MEMORY_DIR", ".agent_memory"))
MEMORY_PATH = MEMORY_DIR / "long_term_memory.json"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    text: str
    source_name: str
    chunk_id: int


def ensure_memory_file() -> None:
    MEMORY_DIR.mkdir(exist_ok=True, parents=True)
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text(json.dumps({"items": []}, indent=2), encoding="utf-8")


def load_long_term_memory() -> List[Dict[str, Any]]:
    ensure_memory_file()
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8")).get("items", [])
    except Exception:
        return []


def save_long_term_memory(items: List[Dict[str, Any]]) -> None:
    ensure_memory_file()
    MEMORY_PATH.write_text(json.dumps({"items": items}, indent=2), encoding="utf-8")


def add_memory_item(text: str, source: str = "user-approved") -> None:
    text = text.strip()
    if not text:
        return
    items = load_long_term_memory()
    items.append({
        "id": f"mem-{int(time.time() * 1000)}",
        "text": text,
        "source": source,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_long_term_memory(items)


def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"\n[Page {page_num}]\n{text}")
    return "\n".join(parts).strip()


def read_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        doc = Document(tmp_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def read_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            pass
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix == ".pdf":
        return read_pdf(data)
    if suffix == ".docx":
        return read_docx(data)
    if suffix in {".txt", ".md", ".py", ".csv", ".json"}:
        return read_text(data)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, source_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, source_name=source_name, chunk_id=idx))
            idx += 1
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    embs = embedder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embs.astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index


def retrieve_docs(query: str, embedder, index, chunks: List[Chunk], top_k: int = 4):
    q_emb = embed_texts(embedder, [query])
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({"score": float(score), "chunk": chunks[int(idx)]})
    return results


def retrieve_memory(query: str, embedder, top_k: int = 3) -> List[Dict[str, Any]]:
    items = load_long_term_memory()
    if not items:
        return []
    texts = [m["text"] for m in items]
    embs = embed_texts(embedder, texts)
    index = build_faiss_index(embs)
    q_emb = embed_texts(embedder, [query])
    scores, indices = index.search(q_emb, min(top_k, len(items)))
    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            item = dict(items[int(idx)])
            item["score"] = float(score)
            hits.append(item)
    return hits


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", ""),
            })
    return results


def fetch_webpage_text(url: str, max_chars: int = 5000) -> str:
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 LocalAgent/1.0"}, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return " ".join(soup.get_text(separator=" ").split())[:max_chars]


def detect_memory_write(question: str) -> Optional[str]:
    patterns = [
        r"remember that (.+)",
        r"note that (.+)",
        r"save this: (.+)",
        r"from now on,? (.+)",
    ]
    for pat in patterns:
        m = re.search(pat, question, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def plan_tools(question: str, has_docs: bool, force_mode: str, allow_memory: bool) -> Dict[str, Any]:
    q = question.lower()
    web_signals = ["latest", "current", "today", "recent", "news", "web", "internet", "search", "who is", "what is happening"]
    doc_signals = ["document", "file", "pdf", "proposal", "uploaded", "report", "according to"]

    if force_mode != "Auto":
        mode = force_mode.lower()
    else:
        use_web = any(sig in q for sig in web_signals)
        use_docs = has_docs and (any(sig in q for sig in doc_signals) or not use_web)
        if has_docs and use_web:
            mode = "hybrid"
        elif use_web:
            mode = "web"
        elif has_docs:
            mode = "docs"
        else:
            mode = "direct"

    memory_candidate = detect_memory_write(question) if allow_memory else None
    actions = []
    if mode in ("docs", "hybrid"):
        actions.append({"name": "retrieve_documents", "description": "Search uploaded document chunks."})
    if mode in ("web", "hybrid"):
        actions.append({"name": "web_search", "description": "Search the public web with DuckDuckGo/ddgs."})
        actions.append({"name": "fetch_top_webpage", "description": "Fetch readable text from the top web result."})
    if allow_memory:
        actions.append({"name": "retrieve_long_term_memory", "description": "Retrieve approved long-term memory relevant to this question."})
    if memory_candidate:
        actions.append({"name": "write_long_term_memory", "description": f"Save this approved memory: {memory_candidate}"})
    actions.append({"name": "generate_answer", "description": "Generate a grounded answer with the local LLM."})
    return {"mode": mode, "actions": actions, "memory_candidate": memory_candidate}


def build_short_term_context(chat_log: List[Tuple[str, str]], max_turns: int = 6) -> str:
    recent = chat_log[-max_turns:]
    return "\n".join([f"{role}: {msg[:1000]}" for role, msg in recent])


def build_prompt(question: str, context_blocks: List[str], short_context: str, memory_blocks: List[str]) -> str:
    context = "\n\n".join(context_blocks) if context_blocks else "No external context."
    memories = "\n".join(memory_blocks) if memory_blocks else "No relevant long-term memory."
    system = (
        "You are a careful local AI agent. Use the provided context and approved memory when useful. "
        "Cite source labels like [Doc 1], [Web 2], or [Memory 1] when you rely on them. "
        "If retrieved context is insufficient, say what is missing. Keep the answer concise."
    )
    return f"""<|system|>
{system}
<|user|>
Short-term conversation context:
{short_context}

Approved long-term memory:
{memories}

Retrieved context:
{context}

Current question:
{question}

Answer:
<|assistant|>
"""


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 400, temperature: float = 0.2) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def ensure_state():
    st.session_state.setdefault("chat_log", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("last_trace", [])


def execute_plan(question: str, plan: Dict[str, Any], settings: Dict[str, Any]):
    trace = [f"Mode: {plan['mode']}"]
    context_blocks: List[str] = []
    memory_blocks: List[str] = []

    embedder = load_embedder(settings["embed_model"])
    tokenizer, model = load_llm(settings["llm_model"])

    if any(a["name"] == "retrieve_documents" for a in plan["actions"]):
        if "index" in st.session_state and "chunks" in st.session_state:
            hits = retrieve_docs(question, embedder, st.session_state["index"], st.session_state["chunks"], settings["top_k"])
            for i, item in enumerate(hits, start=1):
                ch = item["chunk"]
                context_blocks.append(f"[Doc {i}: {ch.source_name} | chunk {ch.chunk_id}]\n{ch.text}")
            trace.append(f"Retrieved {len(hits)} document chunks")

    if any(a["name"] == "web_search" for a in plan["actions"]):
        try:
            hits = web_search(question, settings["web_k"])
            for i, item in enumerate(hits, start=1):
                context_blocks.append(f"[Web {i}: {item['title']} | {item['href']}]\nSnippet: {item['body']}")
            trace.append(f"Retrieved {len(hits)} web results")
            if hits and any(a["name"] == "fetch_top_webpage" for a in plan["actions"]):
                try:
                    page_text = fetch_webpage_text(hits[0]["href"])
                    context_blocks.append(f"[Web page detail: {hits[0]['title']}]\n{page_text}")
                    trace.append("Fetched top web result")
                except Exception as exc:
                    trace.append(f"Top page fetch skipped: {exc}")
        except Exception as exc:
            trace.append(f"Web search failed: {exc}")

    if any(a["name"] == "retrieve_long_term_memory" for a in plan["actions"]):
        mem_hits = retrieve_memory(question, embedder, settings["memory_k"])
        for i, mem in enumerate(mem_hits, start=1):
            memory_blocks.append(f"[Memory {i}] {mem['text']}")
        trace.append(f"Retrieved {len(mem_hits)} long-term memory items")

    if any(a["name"] == "write_long_term_memory" for a in plan["actions"]):
        candidate = plan.get("memory_candidate")
        if candidate:
            add_memory_item(candidate, source="approved-HITL")
            trace.append("Wrote approved item to long-term memory")

    short_context = build_short_term_context(st.session_state["chat_log"])
    prompt = build_prompt(question, context_blocks, short_context, memory_blocks)
    answer = generate_answer(tokenizer, model, prompt, settings["max_tokens"], settings["temperature"])
    trace.append("Generated final answer")
    return answer or "I could not generate an answer.", trace, context_blocks, memory_blocks


ensure_state()
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Local agent with RAG, no-key web search, human-in-the-loop approval, short-term memory, and long-term memory.")

with st.sidebar:
    st.header("Model and Agent Settings")
    model_choice = st.selectbox(
        "LLM model",
        options=[DEFAULT_LLM, DEEPSEEK_LLM, "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
        index=0,
        help="Streamlit Cloud usually works better with the smaller default model. Use DeepSeek locally or on larger compute.",
    )
    custom_model = st.text_input("Custom HF model ID", value="")
    llm_model = custom_model.strip() or model_choice
    embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    mode = st.selectbox("Tool mode", ["Auto", "docs", "web", "hybrid", "direct"])
    hitl = st.checkbox("Human-in-the-loop: approve plans before execution", value=True)
    use_memory = st.checkbox("Use long-term memory", value=True)
    top_k = st.slider("Document chunks", 2, 8, 4)
    web_k = st.slider("Web results", 2, 8, 4)
    memory_k = st.slider("Memory items", 1, 6, 3)
    max_tokens = st.slider("Max new tokens", 128, 1024, 400, 32)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    show_trace = st.checkbox("Show trace and grounding", value=True)
    settings = {
        "llm_model": llm_model,
        "embed_model": embed_model,
        "top_k": top_k,
        "web_k": web_k,
        "memory_k": memory_k,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state["chat_log"] = []
        st.rerun()
    if st.button("Clear document index"):
        for key in ("chunks", "index", "doc_stats"):
            st.session_state.pop(key, None)
        st.rerun()

chat_tab, memory_tab, deploy_tab = st.tabs(["Agent Chat", "Long-Term Memory", "Deploy Notes"])

with memory_tab:
    st.subheader("Long-Term Memory Store")
    items = load_long_term_memory()
    st.write(f"Stored memory items: {len(items)}")
    new_mem = st.text_area("Add a memory item manually", height=100)
    if st.button("Save manual memory") and new_mem.strip():
        add_memory_item(new_mem, source="manual")
        st.success("Saved.")
        st.rerun()
    if items:
        for item in reversed(items[-20:]):
            st.markdown(f"**{item.get('created_at', '')}** — `{item.get('source', '')}`")
            st.write(item["text"])
            st.markdown("---")
    st.download_button("Download memory JSON", json.dumps({"items": items}, indent=2), file_name="long_term_memory.json")
    uploaded_mem = st.file_uploader("Import memory JSON", type=["json"], key="memory_import")
    if uploaded_mem and st.button("Replace memory with uploaded JSON"):
        data = json.loads(uploaded_mem.getvalue().decode("utf-8"))
        save_long_term_memory(data.get("items", []))
        st.success("Memory imported.")
        st.rerun()

with deploy_tab:
    st.subheader("Streamlit Cloud Deployment")
    st.markdown("""
1. Push these files to a public or private GitHub repository.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Use `app.py` as the main file.
4. No API keys or secrets are required.
5. For local/GPU deployment, set `LOCAL_LLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
6. For Streamlit Cloud, keep the smaller default model unless your deployment has enough memory.
""")
    st.code("streamlit run app.py", language="bash")

with chat_tab:
    uploaded_files = st.file_uploader("Upload documents for RAG", type=["pdf", "docx", "txt", "md", "py", "csv", "json"], accept_multiple_files=True)
    if uploaded_files and st.button("Build / Rebuild Document Index", type="primary"):
        chunks: List[Chunk] = []
        stats = []
        with st.spinner("Extracting text and embedding chunks..."):
            embedder = load_embedder(embed_model)
            for f in uploaded_files:
                try:
                    text = extract_text(f)
                    c = chunk_text(text, f.name)
                    chunks.extend(c)
                    stats.append({"file": f.name, "characters": len(text), "chunks": len(c)})
                except Exception as exc:
                    st.error(f"Failed to process {f.name}: {exc}")
            if chunks:
                embs = embed_texts(embedder, [c.text for c in chunks])
                st.session_state["chunks"] = chunks
                st.session_state["index"] = build_faiss_index(embs)
                st.session_state["doc_stats"] = stats
                st.success(f"Indexed {len(chunks)} chunks.")
            else:
                st.warning("No text was indexed.")
    if "doc_stats" in st.session_state:
        with st.expander("Indexed files"):
            st.table(st.session_state["doc_stats"])

    if st.session_state.get("pending"):
        pending = st.session_state["pending"]
        st.warning("Human-in-the-loop approval required before executing the agent plan.")
        st.markdown(f"**Question:** {pending['question']}")
        st.markdown(f"**Planned mode:** `{pending['plan']['mode']}`")
        action_names = [a["name"] for a in pending["plan"]["actions"]]
        edited_actions = st.multiselect(
            "Approve or remove planned actions",
            options=action_names,
            default=action_names,
        )
        for a in pending["plan"]["actions"]:
            st.caption(f"{a['name']}: {a['description']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Approve and run", type="primary"):
                plan = dict(pending["plan"])
                plan["actions"] = [a for a in plan["actions"] if a["name"] in edited_actions]
                with st.spinner("Executing approved plan..."):
                    answer, trace, context_blocks, memory_blocks = execute_plan(pending["question"], plan, pending["settings"])
                st.session_state["chat_log"].append(("user", pending["question"]))
                st.session_state["chat_log"].append(("assistant", answer))
                st.session_state["last_trace"] = trace
                st.session_state["last_context"] = context_blocks
                st.session_state["last_memory"] = memory_blocks
                st.session_state["pending"] = None
                st.rerun()
        with col2:
            if st.button("Deny"):
                st.session_state["pending"] = None
                st.info("Plan denied.")
                st.rerun()
        with col3:
            if st.button("Run without HITL next time"):
                st.session_state["pending"] = None
                st.info("Use the sidebar to turn off HITL.")

    for role, msg in st.session_state["chat_log"]:
        with st.chat_message(role):
            st.write(msg)

    question = st.chat_input("Ask the agent a question")
    if question:
        has_docs = "index" in st.session_state and "chunks" in st.session_state
        plan = plan_tools(question, has_docs, mode, use_memory)
        if hitl:
            st.session_state["pending"] = {"question": question, "plan": plan, "settings": settings}
            st.rerun()
        else:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Agent is executing..."):
                    answer, trace, context_blocks, memory_blocks = execute_plan(question, plan, settings)
                    st.write(answer)
                st.session_state["chat_log"].append(("user", question))
                st.session_state["chat_log"].append(("assistant", answer))
                st.session_state["last_trace"] = trace
                st.session_state["last_context"] = context_blocks
                st.session_state["last_memory"] = memory_blocks

    if show_trace and st.session_state.get("last_trace"):
        with st.expander("Last agent trace and grounding"):
            st.markdown("**Trace**")
            for t in st.session_state.get("last_trace", []):
                st.write("- " + t)
            st.markdown("**Retrieved memory**")
            for m in st.session_state.get("last_memory", []):
                st.text(m[:2000])
            st.markdown("**Retrieved context**")
            for c in st.session_state.get("last_context", []):
                st.text(c[:2500])
                st.markdown("---")
