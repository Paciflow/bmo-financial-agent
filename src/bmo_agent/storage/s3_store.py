from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bmo_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _client(settings: Settings):
    import boto3

    # Only pass explicit keys when we actually have them (local/.env setup). When
    # running inside ECS with a task role attached, there are no explicit keys to
    # pass -- boto3's default credential chain auto-discovers the task role's
    # temporary credentials from the container metadata endpoint. Passing None
    # explicitly would override that and break auth, so we omit the kwargs
    # entirely in that case rather than passing empty strings.
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("s3", **kwargs)


def upload_file(local_path: Path, s3_key: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if not settings.s3_configured:
        raise RuntimeError("S3 is not configured (missing AWS credentials or bucket name).")
    client = _client(settings)
    client.upload_file(str(local_path), settings.s3_bucket_name, s3_key)
    logger.info("Uploaded %s -> s3://%s/%s", local_path, settings.s3_bucket_name, s3_key)
    return s3_key


def upload_json(data: Any, s3_key: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if not settings.s3_configured:
        raise RuntimeError("S3 is not configured (missing AWS credentials or bucket name).")
    client = _client(settings)
    body = json.dumps(data, indent=2).encode("utf-8")
    client.put_object(Bucket=settings.s3_bucket_name, Key=s3_key, Body=body, ContentType="application/json")
    logger.info("Uploaded JSON -> s3://%s/%s", settings.s3_bucket_name, s3_key)
    return s3_key


def check_bucket_access(settings: Settings | None = None) -> bool:
    """Validate S3 credentials/bucket access at startup. Returns True if accessible."""
    settings = settings or get_settings()
    if not settings.s3_configured:
        return False
    try:
        client = _client(settings)
        client.head_bucket(Bucket=settings.s3_bucket_name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 bucket access check failed: %s", exc)
        return False
