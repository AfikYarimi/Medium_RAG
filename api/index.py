from __future__ import annotations

import os
from typing import Optional

# Remove empty OPENAI_BASE_URL before the SDK reads it
if not os.environ.get("OPENAI_BASE_URL", "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

from flask import Flask, request, jsonify
from openai import OpenAI
from pinecone import Pinecone

# ── RAG hyperparameters ───────────────────────────────────────────────────────
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.25
TOP_K = 20

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL      = os.environ.get("CHAT_MODEL", "gpt-5-mini")
INDEX_NAME      = os.environ.get("PINECONE_INDEX_NAME", "medium-articles")

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

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

_openai_client: Optional[OpenAI] = None
_pinecone_index = None


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        kwargs: dict = {"api_key": os.environ["OPENAI_API_KEY"]}
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        _openai_client = OpenAI(**kwargs)
    return _openai_client


def get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


def run_rag(question: str) -> dict:
    client = get_openai()

    embed_res = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    query_vector = embed_res.data[0].embedding

    query_res = get_index().query(vector=query_vector, top_k=TOP_K, include_metadata=True)

    context = []
    for match in query_res.matches or []:
        meta = match.metadata or {}
        context.append({
            "article_id": str(meta.get("article_id", "")),
            "title":      str(meta.get("title", "")),
            "chunk":      str(meta.get("chunk_text", "")),
            "score":      float(match.score or 0.0),
        })

    context_text = "\n\n---\n\n".join(
        f'[{i+1}] Title: "{c["title"]}" | Article ID: {c["article_id"]}\n{c["chunk"]}'
        for i, c in enumerate(context)
    )
    user_prompt = f"Context from Medium articles:\n\n{context_text}\n\n---\n\nQuestion: {question}"

    chat_res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    response = (
        chat_res.choices[0].message.content
        if chat_res.choices
        else "I don't know based on the provided Medium articles data."
    )

    return {
        "response": response,
        "context":  context,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User":   user_prompt,
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/prompt", methods=["POST"])
def prompt_endpoint():
    body = request.get_json(force=True, silent=True) or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return jsonify({"error": 'Missing or empty "question" field.'}), 400
    try:
        return jsonify(run_rag(question))
    except Exception as exc:
        app.logger.error("RAG error: %s", exc, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@app.route("/api/stats", methods=["GET"])
def stats_endpoint():
    return jsonify({
        "chunk_size":    CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k":         TOP_K,
    })


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "description": "Medium Article RAG Assistant",
        "endpoints": {
            "POST /api/prompt": "Query the RAG system",
            "GET /api/stats":   "Get RAG hyperparameter configuration",
        },
    })


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / ".env.local")
    app.run(debug=True, port=5000)
