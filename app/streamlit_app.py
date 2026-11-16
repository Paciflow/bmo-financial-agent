from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = os.environ.get("BMO_AGENT_API_BASE", "http://127.0.0.1:8000")

SUGGESTED_QUESTIONS = [
    "What was BMO's total revenue in fiscal 2025 compared to fiscal 2024?",
    "What is the difference between reported and adjusted net income?",
    "What are the main credit risk factors BMO discusses?",
    "How did the Canadian Personal and Commercial Banking segment perform?",
    "What is a good price-to-earnings ratio for a bank stock?",
    "How did we do this year?",
]

st.set_page_config(page_title="BMO Financial Intelligence Agent", page_icon="📊", layout="wide")


def _api_get(path: str, **kwargs):
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{API_BASE}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()


def _api_post(path: str, json_body: dict, timeout: float = 60.0):
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{API_BASE}{path}", json=json_body)
        return resp


@st.cache_data(ttl=10)
def _health():
    try:
        return _api_get("/health")
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreachable", "error": str(exc)}


@st.cache_data(ttl=10)
def _doc_status():
    try:
        docs = _api_get("/documents")
        return docs[0] if docs else None
    except Exception:  # noqa: BLE001
        return None


def _relevance_label(rerank_score: float | None, rrf_score: float) -> str:
    if rerank_score is not None:
        if rerank_score > 3:
            return "🟢 High relevance"
        if rerank_score > 0:
            return "🟡 Medium relevance"
        return "🔴 Low relevance"
    return f"RRF score: {rrf_score:.4f}"


with st.sidebar:
    st.title("📊 BMO Financial Intelligence Agent")
    st.caption("Grounded Q&A over BMO's 2025 MD&A, with citations and general-knowledge fallback.")

    health = _health()
    status = health.get("status", "unreachable")
    if status == "ok":
        st.success("Backend: healthy")
    elif status == "degraded":
        st.warning("Backend: degraded (indexes missing?)")
    else:
        st.error(f"Backend unreachable at {API_BASE}")

    if status != "unreachable" and not health.get("anthropic_configured", False):
        st.warning("ANTHROPIC_API_KEY not configured on the server — chat will fail until it's set in .env.")

    doc = _doc_status()
    if doc:
        st.subheader("Document status")
        st.write(f"**{doc['document_id']}**")
        st.write(f"Pages: {doc['pages']} | Headings: {doc['headings']}")
        st.write(f"Chunks: {doc['chunks']} | Tables: {doc['tables']}")
        st.write(f"S3 uploaded: {'✅' if doc.get('s3_uploaded') else '❌ (local only)'}")

    show_diagnostics = st.checkbox("Show retrieval diagnostics", value=False)

    st.subheader("Suggested questions")
    for q in SUGGESTED_QUESTIONS:
        if st.button(q, key=f"suggest_{q}", use_container_width=True):
            st.session_state["pending_question"] = q

st.title("Ask about BMO's 2025 MD&A")

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of {question, response}

for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        resp = turn["response"]
        st.write(resp["answer"])

        badge = {
            "document": "📄 Document-grounded",
            "general": "🌐 General knowledge (no citations)",
            "clarification": "❓ Clarifying question",
        }.get(resp.get("response_type"), resp.get("response_type", ""))
        st.caption(badge)

        if resp.get("citations"):
            with st.expander(f"Sources ({len(resp['citations'])})"):
                for c in resp["citations"]:
                    st.markdown(f"**Page {c['page']}** — {c.get('heading_path') or 'N/A'} _(​{c['content_type']})_")
                    st.text(c["excerpt"])
                    if st.button(f"Preview page {c['page']}", key=f"preview_{turn['question']}_{c['chunk_id']}"):
                        try:
                            with httpx.Client(timeout=30.0) as client:
                                img = client.get(f"{API_BASE}/sources/{c['chunk_id']}/page")
                                st.image(img.content, caption=f"PDF page {c['page']}")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Could not load page preview: {exc}")
                    st.divider()

        if show_diagnostics:
            st.json(resp.get("retrieval_diagnostics", {}))

prompt = st.chat_input("Ask a question about BMO's 2025 MD&A, or a general finance question...")
pending = st.session_state.pop("pending_question", None)
question = prompt or pending

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = _api_post("/chat", {"question": question})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
                resp = None

        if resp is not None:
            if resp.status_code == 503:
                st.error(f"Agent not available: {resp.json().get('detail')}")
            elif resp.status_code != 200:
                st.error(f"Backend error ({resp.status_code}): {resp.text}")
            else:
                data = resp.json()
                st.session_state["history"].append({"question": question, "response": data})
                st.rerun()
