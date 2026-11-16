"""End-to-end offline ingestion pipeline: fetch -> extract -> segment -> chunk -> index.

Usage:
    PYTHONPATH=src python scripts/run_ingestion.py [--skip-download] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging

from bmo_agent.chunking.chunker import build_chunks
from bmo_agent.config import get_settings
from bmo_agent.extraction.headings import build_heading_tree
from bmo_agent.extraction.native import extract_native_text
from bmo_agent.extraction.tables import extract_tables
from bmo_agent.indexing.bm25_index import build_bm25_index
from bmo_agent.indexing.qdrant_index import build_section_texts, get_client, index_chunks, index_sections
from bmo_agent.ingestion.fetch import download_pdf
from bmo_agent.embeddings.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_ingestion")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if already ingested")
    args = parser.parse_args()

    settings = get_settings()

    logger.info("Step 1/6: downloading + checksumming BMO PDF (and uploading to S3 if configured)")
    record = download_pdf(settings, force=args.force)
    pdf_path = record.local_path

    logger.info("Step 2/6: native text extraction (PyMuPDF)")
    from pathlib import Path

    pages = extract_native_text(Path(pdf_path), settings)

    logger.info("Step 3/6: heading-based segmentation (embedded PDF TOC)")
    headings = build_heading_tree(Path(pdf_path), settings)

    logger.info("Step 4/6: table structure extraction")
    tables = extract_tables(Path(pdf_path), headings, settings)

    logger.info("Step 5/6: content-aware chunking")
    chunks = build_chunks(pages, headings, tables, settings.document_id, settings)

    logger.info("Step 6/6: dense (Qdrant) + sparse (BM25) indexing")
    embedder = Embedder(settings)
    client = get_client(settings)
    pages_by_index = {p.page_index: p.text for p in pages}
    section_texts = build_section_texts(headings, pages_by_index)
    index_sections(client, headings, section_texts, embedder, settings)
    index_chunks(client, chunks, embedder, settings)
    build_bm25_index(chunks, settings)

    summary = {
        "document_id": settings.document_id,
        "pages": len(pages),
        "headings": len(headings),
        "tables": len(tables),
        "chunks": len(chunks),
        "s3_uploaded": record.s3_uploaded,
    }
    logger.info("Ingestion complete: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
