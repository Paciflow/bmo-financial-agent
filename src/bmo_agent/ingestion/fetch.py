from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from bmo_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class IngestionRecord:
    document_id: str
    source_url: str
    retrieved_at: str
    sha256: str
    size_bytes: int
    local_path: str
    s3_key: str | None
    s3_uploaded: bool
    status: str


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_pdf(settings: Settings | None = None, force: bool = False) -> IngestionRecord:
    """Download the BMO MD&A PDF, validate it, checksum it, and (if configured) upload to S3.

    Idempotent: if a previously-downloaded file with the same checksum already has a
    metadata record, re-download is skipped unless force=True.
    """
    settings = settings or get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    local_path = settings.raw_dir / f"{settings.document_id}.pdf"
    meta_path = settings.raw_dir / f"{settings.document_id}.ingestion.json"

    if meta_path.exists() and local_path.exists() and not force:
        existing = IngestionRecord(**json.loads(meta_path.read_text()))
        if existing.sha256 == _sha256_of(local_path):
            logger.info("PDF already ingested (checksum match); skipping download.")
            return existing

    logger.info("Downloading %s", settings.bmo_pdf_url)
    with httpx.Client(follow_redirects=True, timeout=60.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
        resp = client.get(settings.bmo_pdf_url)
        resp.raise_for_status()
        content = resp.content

    if not content.startswith(b"%PDF-"):
        raise ValueError("Downloaded content does not look like a valid PDF (missing %PDF- header).")

    local_path.write_bytes(content)
    sha256 = _sha256_of(local_path)

    s3_key = None
    s3_uploaded = False
    if settings.s3_configured:
        from bmo_agent.storage.s3_store import upload_file

        s3_key = f"raw/{settings.document_id}/{sha256}.pdf"
        upload_file(local_path, s3_key, settings=settings)
        s3_uploaded = True
    else:
        logger.warning("S3 not configured (missing AWS credentials/bucket) — skipping upload, using local storage only.")

    record = IngestionRecord(
        document_id=settings.document_id,
        source_url=settings.bmo_pdf_url,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        sha256=sha256,
        size_bytes=len(content),
        local_path=str(local_path),
        s3_key=s3_key,
        s3_uploaded=s3_uploaded,
        status="ingested",
    )
    meta_path.write_text(json.dumps(asdict(record), indent=2))
    logger.info("Ingested document_id=%s sha256=%s size=%d s3_uploaded=%s", record.document_id, sha256, record.size_bytes, s3_uploaded)
    return record


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rec = download_pdf()
    print(json.dumps(asdict(rec), indent=2))
