from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Load .env.local when running locally (no-op on Vercel where env vars are injected)
_env = Path(__file__).parent.parent / ".env.local"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)

# Remove empty OPENAI_BASE_URL so the SDK doesn't use "" as a base URL
if not os.environ.get("OPENAI_BASE_URL", "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

from openai import OpenAI
from pinecone import Pinecone

# ── RAG config ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.25
TOP_K = 20
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-5-mini")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-articles")

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    '"I don\'t know based on the provided Medium articles data." '
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant article passage or metadata when helpful."
)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_openai_client: OpenAI | None = None
_pinecone_index = None


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        kwargs: dict = {"api_key": os.environ["OPENAI_API_KEY"]}
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        _openai_client = OpenAI(**kwargs)
    return _openai_client


def _index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def run_rag(question: str) -> dict:
    client = _openai()

    embed_res = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    query_vector = embed_res.data[0].embedding

    query_res = _index().query(vector=query_vector, top_k=TOP_K, include_metadata=True)

    context = []
    for match in query_res.matches or []:
        meta = match.metadata or {}
        context.append({
            "article_id": str(meta.get("article_id", "")),
            "title": str(meta.get("title", "")),
            "chunk": str(meta.get("chunk_text", "")),
            "score": float(match.score or 0.0),
        })

    context_text = "\n\n---\n\n".join(
        f'[{i + 1}] Title: "{c["title"]}" | Article ID: {c["article_id"]}\n{c["chunk"]}'
        for i, c in enumerate(context)
    )
    user_prompt = f"Context from Medium articles:\n\n{context_text}\n\n---\n\nQuestion: {question}"

    chat_res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    response = (
        chat_res.choices[0].message.content
        if chat_res.choices
        else "I don't know based on the provided Medium articles data."
    )

    return {
        "response": response,
        "context": context,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }


# ── Vercel handler ────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            question = str(body.get("question", "")).strip()

            if not question:
                self._respond(400, {"error": 'Missing or empty "question" field.'})
                return

            result = run_rag(question)
            self._respond(200, result)

        except Exception as exc:
            self._respond(500, {"error": str(exc)})

    def _respond(self, status: int, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # suppress default access logs
