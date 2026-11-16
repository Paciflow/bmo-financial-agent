# BMO Financial Intelligence Agent

A document-intelligence agent over BMO Financial Group's 2025 MD&A: hybrid retrieval
(dense + BM25 + RRF + reranking), heading-aware chunking derived from the PDF's own
table of contents, a LangGraph agent that routes between document-grounded answers
(with page citations), general financial knowledge (no fabricated citations), and
clarifying questions for ambiguous queries, served via FastAPI + Streamlit.

## Why this stack (vs. a full AWS Textract/Bedrock/ECS build)

This document (`bmo_ar2025_MDA.pdf`) was inspected before any architecture decisions
were made: **110 pages, 100% native text, zero embedded images** (confirmed via
PyMuPDF during ingestion — see `data/processed/raw_pages.json`). Because of that:

- **No OCR is needed.** Amazon Textract exists for scanned/image-based documents;
  this PDF isn't one. Native extraction with PyMuPDF is complete and more reliable.
- **Heading segmentation uses the PDF's own embedded table of contents** (55 entries,
  2 levels, exact page ranges) rather than inferring structure from font sizes or an
  LLM — this is exact, auditable, and free.
- **Table structure** comes from PyMuPDF's built-in table detector. Cell text is
  cleaned for readability with dictionary-based word segmentation (see
  `chunking/text_cleanup.py`) rather than an LLM pass — it turned out to fully fix
  the retrieval-quality problem (see evaluation results below) without the added
  cost/latency of a vision-model call per table. The original structural extraction
  is always preserved untouched in `data/processed/tables.json` for audit, per the
  "never silently overwrite extracted data" requirement. `interpretation/` is a
  stub for a future Bedrock/Claude-vision pass on page-image crops, which would
  still be worth adding for the handful of tables with genuinely ambiguous layout.
- **Embeddings are local** (`sentence-transformers`), so dense retrieval has zero
  network dependency at query time — important for a live demo.
- **The LLM is Claude** (Anthropic API) rather than Bedrock, to avoid gating a
  live demo on Bedrock model-access approval latency.
- **S3 is real** (if configured) for document storage/provenance; Bedrock/Textract/
  ECS Fargate remain documented as the production target architecture (see
  `slides/`) rather than live dependencies for this prototype.

Everything here is swappable: `embeddings/embedder.py` and `agent/llm.py` are the
two seams where Bedrock Titan Embeddings and Bedrock-hosted Claude would plug in for
a production deployment without changing any other module.

## Project layout

```
src/bmo_agent/
  ingestion/        download + checksum + S3 upload, idempotent
  extraction/        PyMuPDF native text, TOC-based headings, table structure
  chunking/           content-aware chunking (narrative 1200/200, table row-groups)
  interpretation/    vision-based diagram interpretation (Claude + rendered page images)
  chunking/text_cleanup.py  word-segmentation cleanup for glued table label text
  validation/         numeric/structural consistency checks
  embeddings/         local sentence-transformers wrapper
  indexing/            Qdrant (embedded, local) + BM25
  retrieval/          hybrid search, RRF, cross-encoder rerank, table expansion
  agent/               LangGraph orchestration, prompts, deterministic calculator
  api/                 FastAPI (the only thing Streamlit talks to)
  eval/                 20 ground-truth questions + retrieval/answer evaluation
app/streamlit_app.py  Streamlit frontend (talks to FastAPI only)
scripts/run_ingestion.py  single command for the whole offline pipeline
```

## Setup

Requires Python 3.11 (this repo's venv was built with Homebrew's `python3.11`;
a newer system Python may have compatibility issues with some pinned libraries).

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env
# edit .env: at minimum set ANTHROPIC_API_KEY; set AWS_* + S3_BUCKET_NAME for real S3 storage
```

## Running the offline ingestion pipeline

One command runs the entire offline path: download → checksum → (S3 upload if
configured) → native extraction → heading segmentation → table extraction →
chunking → dense (Qdrant) + sparse (BM25) indexing.

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_ingestion.py
```

It's idempotent — re-running without `--force` skips re-download if the checksum
already matches what's on disk.

Optionally, after ingestion, add vision-based interpretation of the document's
diagram content (requires `ANTHROPIC_API_KEY`) and re-index to pick it up:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/interpret_diagrams.py
PYTHONPATH=src ./.venv/bin/python -m bmo_agent.indexing.bm25_index
PYTHONPATH=src ./.venv/bin/python -m bmo_agent.indexing.qdrant_index
```

This must run *after* `run_ingestion.py`, since re-running the main pipeline
regenerates `chunks.json` from scratch and would drop the diagram chunk.

## Running the app

```bash
# backend
PYTHONPATH=src ./.venv/bin/uvicorn bmo_agent.api.main:app --port 8000

# frontend (separate terminal)
./.venv/bin/streamlit run app/streamlit_app.py
```

`GET /health` reports whether indexes, the Anthropic key, and S3 are configured. If
`ANTHROPIC_API_KEY` is missing, `/chat` returns a clear `503` rather than a stack
trace — the rest of the API (`/documents`, `/retrieve`, `/sources/*`) works without
it, since those are retrieval-only.

### Running it via Docker instead

`docker/Dockerfile` and `docker/docker-compose.yml` build the same two services
(API + Streamlit) as containers. **Written but not build-tested in this
environment** (Docker isn't installed here) — review before relying on it for a
live demo; running the two `./.venv/bin/...` commands above directly is the
verified path.

```bash
# from the project root, after running ingestion at least once locally so
# data/processed, data/bm25, data/qdrant exist to copy into the shared volume
docker compose -f docker/docker-compose.yml up --build
```

Both containers share one named volume (`bmo_data`) mounted at `/app/data`, since
Qdrant's embedded mode needs the same files both services were indexed into
visible to whichever container is serving requests. Ingestion itself isn't
containerized as a separate one-shot job yet (a future iteration could run that
as its own ECS Fargate task) — run it locally first, or `docker compose exec
api python scripts/run_ingestion.py` once the `api` container is up.

## Evaluation

Two scripts, split because one needs an LLM key and one doesn't:

```bash
# Retrieval-only: Recall@5 and MRR for dense-only / BM25-only / hybrid / hybrid+rerank.
# No API key required.
PYTHONPATH=src ./.venv/bin/python -m bmo_agent.eval.run_retrieval_eval

# End-to-end: citation accuracy, numeric accuracy, clarification routing,
# no-fabricated-citation rate, latency. Requires ANTHROPIC_API_KEY.
PYTHONPATH=src ./.venv/bin/python -m bmo_agent.eval.run_answer_eval
```

20 questions (`src/bmo_agent/eval/questions.py`) span narrative retrieval, tables,
reported-vs-adjusted figures, YoY comparisons, operating segments, ambiguous
questions, general-knowledge questions, and unsupported questions — each with
manually verified expected page citations (confirmed by reading the source page
before writing the question, not generated by an LLM). Results are written to
`data/processed/eval/*.json`.

**Latest measured retrieval results** (see `data/processed/eval/retrieval_eval_summary.json`
for the current run — do not trust this table once the pipeline changes without
re-running):

| Mode | Recall@5 | MRR | Avg latency |
|---|---|---|---|
| dense-only | 0.571 | 0.340 | ~12 ms |
| BM25-only | 0.643 | 0.366 | <1 ms |
| hybrid (RRF) | 0.643 | 0.395 | ~6 ms |
| hybrid + rerank | 0.714 | 0.502 | ~140 ms |

**End-to-end answer evaluation** (`data/processed/eval/answer_eval_summary.json`,
real Anthropic API calls, 20 questions): citation accuracy 0.57, clarification
routing accuracy 1.00 (both ambiguous questions correctly triggered a clarifying
question instead of guessing), no-fabricated-citation rate 1.00 (general-knowledge
and unsupported questions never attached a BMO-document citation they didn't earn),
average latency ~8.3s/question.

These numbers came from real, iterative debugging, not a first-try pipeline:
- Hybrid+rerank improved materially (Recall@5 0.643→0.714, MRR 0.477→0.502) after
  fixing a table-extraction artifact where PyMuPDF's table reader concatenated label
  words with no spaces (`"Netinterestincome"`), which meant BM25 could never match a
  query term like "interest" against it. See `chunking/text_cleanup.py`.
- A second bug meant a fixed 3-row table header absorbed real data rows, causing
  "Adjusted Results" figures to carry a leftover "Reported Results" tag in later
  chunks — fixed with dynamic header detection (`_detect_header_row_count`).
- citation_accuracy spot-checks show the *numbers* the model states are consistently
  correct (verified by hand against `expected_values`); most misses are a citation
  landing on the page immediately before/after the expected one, where a chunk spans
  a page boundary and the chunk is tagged with its first page rather than the exact
  page a given sentence falls on — a real precision limitation worth fixing before
  a production deployment, not a fabrication.
- A citation-honesty bug was also fixed: an "I can't find this in the evidence"
  answer used to still attach citations from whatever (irrelevant) passages were
  retrieved. Citations are now filtered to only pages the answer actually cites
  inline (see `agent/graph.py::_referenced_pages`).

BM25 alone is surprisingly strong on this document because financial terminology
and table labels are highly literal — a finding worth highlighting rather than
hiding.

## Tests

```bash
./.venv/bin/python -m pytest tests/ -v
```

Covers RRF fusion math, deterministic financial calculations (including the
percent-change vs. percentage-point-change distinction), the BM25 tokenizer, and
content-aware chunking (boundary respect, overlap, table row-group splitting).

## Known limitations (honest, not hidden)

- PyMuPDF's table extraction on this PDF's densest tables originally glued label
  words together with no spaces (`"Netinterestincome"`). `chunking/text_cleanup.py`
  fixes this with dictionary-based word segmentation (`wordninja`) applied only to
  tokens containing letters, never to numeric tokens (so a negative value like
  `(164)` is never touched). It's best-effort, not perfect — uncommon words
  (e.g. "divestitures") can still mis-segment. The original, untouched extraction
  stays in `data/processed/tables.json` for audit; only the rendered retrieval text
  is cleaned.
- A single table's repeated chunk header can go stale across a Reported→Adjusted
  section transition within the same table (the data itself is correct in the
  chunk body, just the repeated header tag doesn't track the transition) — see
  `README`'s evaluation section above for the specific case found during testing.
- Citations occasionally land on the page immediately before/after the page a cited
  sentence is physically on, when a chunk spans a page boundary (the chunk is
  tagged with its first page). The stated figures are correct; the page number can
  be off by one in that specific scenario.
- This document has zero embedded *raster* images, but it does have vector-drawn
  diagrams (box/line shapes drawn directly in the PDF, not image XObjects) — found
  by noticing the agent's own answer for a segment-structure question flagged its
  evidence as "garbled, columns running together" (native text extraction linearizes
  a 2D box diagram into a flat word sequence). `interpretation/diagrams.py` renders
  a page to an image and asks Claude's vision capability to describe it instead —
  tested on BMO's operating-segments org chart (page 20) and it correctly recovered
  all four segments and their lines of business. Reliably auto-detecting *which*
  pages need this (vector-drawing count alone doesn't separate a diagram from a
  heavily-gridded table — BMO's statistical tables use just as many ruled lines) is
  future work; `scripts/interpret_diagrams.py` currently runs against a short,
  manually-reviewed page list rather than a classifier.
- Dense embeddings use a small general-purpose local model
  (`all-MiniLM-L6-v2`) for zero-dependency reliability; a production deployment
  would swap in Bedrock Titan Text Embeddings V2 or a finance-tuned model.
- Short, single-mention facts inside a longer overview paragraph (e.g. "total
  assets $1.5 trillion" inside the one-paragraph "About BMO" section) can lose
  to chunks that repeat the same words many times in an unrelated table, because
  both BM25 and dense search are influenced by in-chunk term frequency. A
  section-level retrieval boost (dense-matching the query against the 55 section
  summaries already indexed in the `bmo_sections` collection, then boosting
  chunks from the top-matched section) was built and tested, and it did fix the
  "About BMO" case -- but it also made "About BMO" rank #1 for an unrelated
  "Capital Markets revenue" query, because `all-MiniLM-L6-v2` scored "About BMO"
  as *more* semantically similar to that query (0.564) than the actual Capital
  Markets section (which didn't even place in the top 3). That's a worse
  regression than the problem it fixed, so it was reverted rather than shipped.
  Revisiting this with a better section-matching model (or BM25 over section
  titles instead of dense embeddings) is the right next step, not a blanket
  score threshold tweak.

## Production AWS deployment (target architecture, not deployed by this repo)

See `slides/` for the full target architecture diagram. Summary: S3 (raw PDFs +
processed JSON/Markdown), Textract (only if a future source document is scanned),
Bedrock (Claude + Titan Embeddings V2, swapping in for the local equivalents behind
the same two interfaces), Qdrant on a persistent volume (ECS task or EC2), ECS
Fargate services for the ingestion job, FastAPI, and Streamlit, CloudWatch for logs,
Secrets Manager for credentials, and IAM least-privilege roles per service — no
hardcoded AWS keys.
