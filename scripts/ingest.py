"""
Ingestion script: reads the Medium CSV, chunks each article,
embeds with text-embedding-3-small, and upserts to Pinecone.

Usage:
  python scripts/ingest.py                  # full corpus
  python scripts/ingest.py --limit 200      # first 200 articles (for testing)
  python scripts/ingest.py --offset 500     # start from article 500

Make sure .env.local is filled in before running.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.local")

# Remove empty OPENAI_BASE_URL so the SDK doesn't treat "" as the base URL.
if not os.environ.get("OPENAI_BASE_URL", "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

from openai import OpenAI  # noqa: E402 (after dotenv load)
from pinecone import Pinecone, ServerlessSpec  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-articles")
DIMENSIONS = 1536

CHUNK_WORDS = 384        # ≈512 tokens at 0.75 words/token
OVERLAP_RATIO = 0.25
OVERLAP_WORDS = int(OVERLAP_RATIO * CHUNK_WORDS)       # 25% of 384 = 96 words
STEP = CHUNK_WORDS - OVERLAP_WORDS

EMBED_BATCH = 50         # texts per embedding API call
UPSERT_BATCH = 100       # vectors per Pinecone upsert call
FLUSH_EVERY = 300        # flush accumulated vectors every N chunks

CSV_PATH = Path(__file__).parent.parent / "medium-english-50mb.csv"

# ── Clients ───────────────────────────────────────────────────────────────────
_kwargs: dict = {"api_key": os.environ["OPENAI_API_KEY"]}
if os.environ.get("OPENAI_BASE_URL", "").strip():
    _kwargs["base_url"] = os.environ["OPENAI_BASE_URL"].strip()
openai_client = OpenAI(**_kwargs)

pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        if chunk:
            chunks.append(chunk)
        if i + CHUNK_WORDS >= len(words):
            break
        i += STEP
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    for attempt in range(4):
        try:
            res = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [d.embedding for d in res.data]
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (429, 503) and attempt < 3:
                wait = 5 * (attempt + 1)
                print(f"  Rate limit/server error, retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after retries")


def ensure_index():
    existing = {idx.name for idx in (pinecone_client.list_indexes().indexes or [])}
    if INDEX_NAME not in existing:
        print(f'Creating Pinecone index "{INDEX_NAME}"…')
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Waiting 30s for index to become ready…")
        time.sleep(30)
    else:
        print(f'Index "{INDEX_NAME}" already exists.')


def upsert_vectors(index, vectors: list[dict]):
    for i in range(0, len(vectors), UPSERT_BATCH):
        index.upsert(vectors=vectors[i : i + UPSERT_BATCH])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max articles to ingest")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N articles")
    args = parser.parse_args()

    ensure_index()
    index = pinecone_client.Index(INDEX_NAME)

    pending_ids: list[str] = []
    pending_texts: list[str] = []
    pending_meta: list[dict] = []

    article_count = 0
    total_chunks = 0

    def flush():
        nonlocal total_chunks
        if not pending_texts:
            return
        print(f"  Embedding {len(pending_texts)} chunks…")
        for i in range(0, len(pending_texts), EMBED_BATCH):
            batch_ids = pending_ids[i : i + EMBED_BATCH]
            batch_texts = pending_texts[i : i + EMBED_BATCH]
            batch_meta = pending_meta[i : i + EMBED_BATCH]
            embeddings = embed_texts(batch_texts)
            vectors = [
                {"id": vid, "values": emb, "metadata": meta}
                for vid, emb, meta in zip(batch_ids, embeddings, batch_meta)
            ]
            upsert_vectors(index, vectors)
        total_chunks += len(pending_texts)
        pending_ids.clear()
        pending_texts.clear()
        pending_meta.clear()
        print(f"  Flushed. Total chunks so far: {total_chunks}")

    print(f"Reading CSV from {CSV_PATH}")
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if row_idx < args.offset:
                continue
            if args.limit is not None and article_count >= args.limit:
                break

            text = (row.get("text") or "").strip()
            if not text:
                article_count += 1
                continue

            title = (row.get("title") or "").strip()
            url = (row.get("url") or "").strip()
            authors = (row.get("authors") or "").strip()
            article_id = str(row_idx)

            chunks = chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                pending_ids.append(f"article_{article_id}_chunk_{chunk_idx}")
                pending_texts.append(chunk)
                pending_meta.append(
                    {
                        "article_id": article_id,
                        "title": title,
                        "chunk_text": chunk,
                        "url": url,
                        "authors": authors,
                    }
                )

            article_count += 1
            if article_count % 100 == 0:
                print(f"Processed {article_count} articles, {len(pending_texts)} chunks pending…")

            if len(pending_texts) >= FLUSH_EVERY:
                flush()

    flush()
    print(f"\nDone! Ingested {article_count} articles, {total_chunks} total chunks.")


if __name__ == "__main__":
    main()
