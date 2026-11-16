FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY docker/start.sh docker/start.sh
RUN chmod +x docker/start.sh

# The pre-built index, baked into the image -- this is the scoped "cloud
# deployment" decision: no separate cloud vector-DB service, Qdrant stays in
# its embedded/file-based mode, just local-to-the-container instead of
# local-to-a-laptop. Rebuild and redeploy the image to pick up a re-ingested
# document; there's no live ingestion step at container startup.
COPY data/raw/ data/raw/
COPY data/processed/ data/processed/
COPY data/qdrant/ data/qdrant/
COPY data/bm25/ data/bm25/

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
# Models are baked into the image at build time (below); without this, the
# huggingface_hub library still phones home on every load to check for
# updates/metadata even though the weights are already cached locally --
# confirmed in testing (GET https://huggingface.co/api/models/... on every
# startup). That's exactly the live network dependency the local-embeddings
# design was meant to avoid; this forces cache-only, no network, ever.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Pre-download the local embedding + reranking models at build time so the
# container's first request isn't the one paying for the download.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-base-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8501

CMD ["bash", "docker/start.sh"]
