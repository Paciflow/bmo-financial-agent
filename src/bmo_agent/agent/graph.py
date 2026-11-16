from __future__ import annotations

import logging
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from bmo_agent.agent import prompts
from bmo_agent.agent.calculations import compute
from bmo_agent.agent.llm import complete, complete_structured
from bmo_agent.config import Settings, get_settings
from bmo_agent.retrieval.hybrid import EvidencePassage, HybridRetriever

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    question: str
    category: str
    search_query: str
    clarifying_question: str | None
    evidence: list[EvidencePassage]
    retrieval_attempts: int
    sufficient: bool
    verification_reason: str
    computed_figure: dict | None
    answer: str
    citations: list[dict]
    response_type: str
    retrieval_diagnostics: dict
    timings: dict


def _format_evidence(evidence: list[EvidencePassage]) -> str:
    parts = []
    for e in evidence:
        parts.append(
            f"[chunk_id={e.chunk_id} page={e.page_index + 1} section=\"{e.heading_path or 'N/A'}\" "
            f"type={e.content_type}]\n{e.text}"
        )
    return "\n\n---\n\n".join(parts)


_PAGE_CITE_RE = re.compile(r"\bpp?\.\s*(\d+(?:\s*,\s*\d+)*)", re.IGNORECASE)


def _referenced_pages(answer_text: str) -> set[int]:
    """Pages the model actually wrote an inline citation for -- "p. 24", "pp. 10, 41",
    or "p." appearing anywhere in a parenthetical, not just immediately after the
    opening paren (e.g. "(Table 21, p. 38)" must still be caught; an earlier version
    of this regex required "(p." literally at the start of the parens and silently
    dropped citations like that one, confirmed during benchmark review). Used to
    filter the citation list down to pages the answer truly relies on -- otherwise a
    "the evidence doesn't contain this" answer would still list retrieved-but-unused
    pages as if they supported it.
    """
    pages: set[int] = set()
    for match in _PAGE_CITE_RE.finditer(answer_text):
        for num in match.group(1).split(","):
            num = num.strip()
            if num.isdigit():
                pages.add(int(num))
    return pages


def _citations_from_evidence(evidence: list[EvidencePassage]) -> list[dict]:
    seen_pages: set[int] = set()
    citations = []
    for e in evidence:
        if e.page_index in seen_pages:
            continue
        seen_pages.add(e.page_index)
        citations.append(
            {
                "chunk_id": e.chunk_id,
                "page": e.page_index + 1,
                "heading_path": e.heading_path,
                "content_type": e.content_type,
                "excerpt": e.text[:300],
            }
        )
    return citations


class BMOAgent:
    def __init__(self, settings: Settings | None = None, retriever: HybridRetriever | None = None):
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(self.settings)
        self.graph = self._build_graph()

    # ---- nodes ----

    def _classify(self, state: AgentState) -> AgentState:
        try:
            parsed = complete_structured(
                prompts.SYSTEM_PROMPT,
                prompts.CLASSIFY_PROMPT.format(question=state["question"]),
                tool_name="classify_question",
                tool_description="Classify the user's question and extract a retrieval query.",
                input_schema=prompts.CLASSIFY_SCHEMA,
                settings=self.settings,
                max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("classify: tool call failed, defaulting to 'document'. error=%r", exc)
            parsed = {"category": "document", "search_query": state["question"], "clarifying_question": None}
        return {
            **state,
            "category": parsed.get("category", "document"),
            "search_query": parsed.get("search_query") or state["question"],
            "clarifying_question": parsed.get("clarifying_question"),
        }

    def _route_after_classify(self, state: AgentState) -> str:
        return {
            "ambiguous": "clarify",
            "general": "answer_general",
            "document": "retrieve",
            "calculation": "retrieve",
        }.get(state["category"], "retrieve")

    def _clarify(self, state: AgentState) -> AgentState:
        question = state.get("clarifying_question") or (
            "Could you clarify which operating segment, time period, or metric you're asking about?"
        )
        return {**state, "answer": question, "response_type": "clarification", "citations": []}

    def _retrieve(self, state: AgentState) -> AgentState:
        attempts = state.get("retrieval_attempts", 0) + 1
        result = self.retriever.retrieve(state.get("search_query") or state["question"])
        return {**state, "evidence": result.evidence, "retrieval_attempts": attempts, "retrieval_diagnostics": result.diagnostics}

    def _verify(self, state: AgentState) -> AgentState:
        evidence_text = _format_evidence(state.get("evidence", []))
        try:
            parsed = complete_structured(
                prompts.SYSTEM_PROMPT,
                prompts.VERIFY_EVIDENCE_PROMPT.format(question=state["question"], evidence=evidence_text),
                tool_name="verify_evidence",
                tool_description="Judge whether the retrieved evidence is sufficient to answer the question.",
                input_schema=prompts.VERIFY_EVIDENCE_SCHEMA,
                settings=self.settings,
                max_tokens=200,
            )
            sufficient = bool(parsed.get("sufficient", True))
            reason = parsed.get("reason", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("verify: tool call failed, assuming sufficient. error=%r", exc)
            sufficient, reason = True, "verification call failed; proceeding with best-effort answer"
        return {**state, "sufficient": sufficient, "verification_reason": reason}

    def _route_after_verify(self, state: AgentState) -> str:
        if state.get("sufficient"):
            return "extract_figures" if state["category"] == "calculation" else "answer_document"
        if state.get("retrieval_attempts", 0) < MAX_RETRIEVAL_ATTEMPTS:
            return "retrieve"
        return "answer_document"  # give best-effort answer with an explicit insufficiency caveat

    def _extract_figures(self, state: AgentState) -> AgentState:
        evidence_text = _format_evidence(state.get("evidence", []))
        computed = None
        try:
            parsed = complete_structured(
                prompts.SYSTEM_PROMPT,
                prompts.EXTRACT_FIGURES_PROMPT.format(question=state["question"], evidence=evidence_text),
                tool_name="extract_figures",
                tool_description="Identify the two figures and operation needed for a financial calculation.",
                input_schema=prompts.EXTRACT_FIGURES_SCHEMA,
                settings=self.settings,
                max_tokens=400,
            )
            if parsed.get("found"):
                a, b = parsed["figure_a"], parsed["figure_b"]
                result = compute(parsed["operation"], float(a["value"]), float(b["value"]))
                computed = {"figure_a": a, "figure_b": b, "operation": parsed["operation"], **result}
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract_figures failed: %s", exc)
        return {**state, "computed_figure": computed}

    def _answer_document(self, state: AgentState) -> AgentState:
        evidence = state.get("evidence", [])
        evidence_text = _format_evidence(evidence)
        extra = ""
        if state.get("computed_figure"):
            cf = state["computed_figure"]
            extra = (
                f"\n\nA deterministic calculation has already been performed -- state this result verbatim, "
                f"do not recompute it: {cf['figure_a']['label']} = {cf['figure_a']['value']}, "
                f"{cf['figure_b']['label']} = {cf['figure_b']['value']}, operation={cf['operation']}, "
                f"result={cf['result']} {cf.get('unit') or ''} (formula: {cf['formula']})."
            )
        if not state.get("sufficient", True):
            extra += "\n\nNote: retrieved evidence was judged insufficient after retries -- say explicitly what is missing."

        prompt = prompts.ANSWER_DOCUMENT_PROMPT.format(evidence=evidence_text, question=state["question"]) + extra
        answer = complete(prompts.SYSTEM_PROMPT, prompt, self.settings, max_tokens=1024)

        all_citations = _citations_from_evidence(evidence)
        referenced = _referenced_pages(answer)
        # Only keep citations for pages the answer actually cited inline. The
        # prompt requires every factual claim to carry a "(p. N)" citation, so an
        # answer with none means it didn't ground itself in the evidence (e.g. "the
        # evidence doesn't contain this") -- it must not still list retrieved-but-
        # unused pages as if they backed up an answer that was never given.
        citations = [c for c in all_citations if c["page"] in referenced]

        return {
            **state,
            "answer": answer,
            "citations": citations,
            "response_type": "document",
        }

    def _answer_general(self, state: AgentState) -> AgentState:
        answer = complete(
            prompts.SYSTEM_PROMPT,
            prompts.ANSWER_GENERAL_PROMPT.format(question=state["question"]),
            self.settings,
            max_tokens=700,
        )
        return {**state, "answer": answer, "citations": [], "response_type": "general"}

    # ---- graph assembly ----

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("classify", self._classify)
        g.add_node("clarify", self._clarify)
        g.add_node("retrieve", self._retrieve)
        g.add_node("verify", self._verify)
        g.add_node("extract_figures", self._extract_figures)
        g.add_node("answer_document", self._answer_document)
        g.add_node("answer_general", self._answer_general)

        g.set_entry_point("classify")
        g.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {"clarify": "clarify", "answer_general": "answer_general", "retrieve": "retrieve"},
        )
        g.add_edge("retrieve", "verify")
        g.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"retrieve": "retrieve", "answer_document": "answer_document", "extract_figures": "extract_figures"},
        )
        g.add_edge("extract_figures", "answer_document")
        g.add_edge("clarify", END)
        g.add_edge("answer_document", END)
        g.add_edge("answer_general", END)
        return g.compile()

    def ask(self, question: str) -> dict[str, Any]:
        start = time.perf_counter()
        final_state = self.graph.invoke({"question": question, "retrieval_attempts": 0})
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "answer": final_state.get("answer", ""),
            "response_type": final_state.get("response_type", "unknown"),
            "citations": final_state.get("citations", []),
            "category": final_state.get("category"),
            "retrieval_diagnostics": final_state.get("retrieval_diagnostics", {}),
            "latency_ms": round(latency_ms, 1),
        }
