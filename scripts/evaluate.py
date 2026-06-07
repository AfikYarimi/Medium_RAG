"""
Hyperparameter evaluation for the Medium RAG system.

Evaluates two things:
  1. top_k sweep (no re-embedding needed — just queries Pinecone)
  2. chunk_size x overlap_ratio grid (re-embeds a small 200-article subset)

Metrics per configuration:
  - avg_score      : mean cosine similarity of retrieved chunks (higher = more relevant)
  - diversity      : fraction of top-k chunks that come from distinct articles (higher = better
                     for multi-result queries)
  - answer_quality : 1-5 rating from gpt-5-mini acting as a judge (higher = better answer)

Usage:
  python scripts/evaluate.py --mode topk        # fast, no extra embedding cost
  python scripts/evaluate.py --mode chunks      # slow, re-embeds 200-article subset
  python scripts/evaluate.py --mode all         # both
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.local")
if not os.environ.get("OPENAI_BASE_URL", "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc     = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-5-mini"
MAIN_INDEX      = os.environ.get("PINECONE_INDEX_NAME", "medium-articles")

# ── Test questions (one per query type) ──────────────────────────────────────

TEST_QUESTIONS = [
    "Find an article that reframes marketing as a conversation with readers, "
    "aimed at writers who find self-promotion uncomfortable. Provide title and author.",

    "List exactly 3 articles about education. Return only the titles.",

    "Find an article that argues past pandemics can spur innovation and recovery, "
    "and summarise its central argument.",

    "I want practical, beginner-friendly advice on building habits that actually stick. "
    "Which article would you recommend, and why?",
]

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you. "
    "If the answer cannot be determined from the provided context, respond: "
    '"I don\'t know based on the provided Medium articles data."'
)

JUDGE_PROMPT = (
    "Rate the following RAG answer on a scale of 1-5 for relevance and completeness "
    "given the question. Reply with ONLY a single integer (1-5).\n\n"
    "Question: {question}\n\nAnswer: {answer}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBEDDING_MODEL, input=text).data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    res = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in res.data]


def query_index(index_name: str, vector: list[float], top_k: int) -> list:
    idx = pc.Index(index_name)
    return idx.query(vector=vector, top_k=top_k, include_metadata=True).matches or []


def build_context(matches: list) -> str:
    return "\n\n---\n\n".join(
        f'[{i+1}] Title: "{m.metadata.get("title","")}" | ID: {m.metadata.get("article_id","")}\n'
        f'{m.metadata.get("chunk_text","")}'
        for i, m in enumerate(matches)
    )


def ask(index_name: str, question: str, top_k: int) -> tuple[str, list]:
    vec     = embed(question)
    matches = query_index(index_name, vec, top_k)
    ctx     = build_context(matches)
    user_p  = f"Context from Medium articles:\n\n{ctx}\n\n---\n\nQuestion: {question}"
    answer  = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_p},
        ],
    ).choices[0].message.content
    return answer, matches


def judge(question: str, answer: str) -> int:
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    raw = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content.strip()
    try:
        return max(1, min(5, int(raw[0])))
    except Exception:
        return 3


def metrics(matches: list, top_k: int) -> dict:
    scores   = [m.score for m in matches if m.score is not None]
    art_ids  = [m.metadata.get("article_id", "") for m in matches]
    unique   = len(set(art_ids))
    return {
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "diversity": round(unique / max(len(matches), 1), 3),
    }


def format_table(rows: list[dict], title: str) -> str:
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]
    if not rows:
        return "\n".join(lines)
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append("  ".join(str(r[k]).ljust(widths[k]) for k in keys))
    return "\n".join(lines)


def print_table(rows: list[dict], title: str):
    print(format_table(rows, title))


def save_report(sections: list[tuple[str, list[dict], str]], out_path: Path):
    """Write a human-readable report of all evaluation results to a .txt file."""
    import datetime
    lines = [
        "Medium RAG — Hyperparameter Evaluation Report",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Metrics explained:",
        "  avg_score      – mean cosine similarity of retrieved chunks (higher = more relevant)",
        "  diversity      – fraction of top-k from distinct articles  (higher = better for multi-result queries)",
        "  quality(1-5)   – gpt-5-mini self-judge score averaged over 4 test questions",
        "",
        "Test questions used:",
    ]
    for i, q in enumerate(TEST_QUESTIONS, 1):
        lines.append(f"  Q{i}: {q}")
    lines.append("")

    for label, rows, title in sections:
        lines.append(format_table(rows, title))
        lines.append("")

        if rows:
            # Highlight best row per numeric metric
            numeric_keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))
                            and k not in ("chunk_size", "top_k", "overlap", "n_chunks")]
            lines.append("  Best settings per metric:")
            for k in numeric_keys:
                best = max(rows, key=lambda r: r[k])
                lines.append(f"    {k}: chunk_size={best.get('chunk_size','–')} "
                             f"overlap={best.get('overlap','–')} → {best[k]}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report saved to: {out_path}")


# ── Mode 1: top_k sweep (no re-embedding) ────────────────────────────────────

def eval_topk():
    print("\n[top_k sweep] Using existing index, no re-embedding needed.")
    TOP_K_VALUES = [3, 5, 10, 15, 20, 30]
    results = []

    for top_k in TOP_K_VALUES:
        scores_all, div_all, quality_all = [], [], []
        for q in TEST_QUESTIONS:
            answer, matches = ask(MAIN_INDEX, q, top_k)
            m = metrics(matches, top_k)
            scores_all.append(m["avg_score"])
            div_all.append(m["diversity"])
            quality_all.append(judge(q, answer))
            time.sleep(0.3)

        results.append({
            "top_k":        top_k,
            "avg_score":    round(sum(scores_all)  / len(scores_all),  4),
            "diversity":    round(sum(div_all)      / len(div_all),     3),
            "answer_quality(1-5)": round(sum(quality_all) / len(quality_all), 2),
        })
        print(f"  top_k={top_k} done.")

    print_table(results, "top_k sweep results (chunk_size=512, overlap=0.2)")
    return results


# ── Mode 2: chunk_size x overlap grid (re-embeds 200-article subset) ─────────

def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    step  = chunk_words - overlap_words
    chunks, i = [], 0
    while i < len(words):
        c = " ".join(words[i : i + chunk_words])
        if c:
            chunks.append(c)
        if i + chunk_words >= len(words):
            break
        i += step
    return chunks


def build_temp_index(name: str, vectors: list[dict]):
    existing = {idx.name for idx in (pc.list_indexes().indexes or [])}
    if name in existing:
        pc.delete_index(name)
        time.sleep(5)
    pc.create_index(
        name=name, dimension=1536, metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(30)
    idx = pc.Index(name)
    for i in range(0, len(vectors), 100):
        idx.upsert(vectors=vectors[i : i + 100])
    time.sleep(5)
    return idx


def eval_chunks(subset: int = 200):
    print(f"\n[chunk grid] Re-embedding first {subset} articles for each config.")
    print("  WARNING: this costs ~$0.01-0.05 in OpenAI credits per config.")

    CONFIGS = [
        {"chunk_size": 256, "overlap_ratio": 0.1},
        {"chunk_size": 256, "overlap_ratio": 0.2},
        {"chunk_size": 512, "overlap_ratio": 0.1},
        {"chunk_size": 512, "overlap_ratio": 0.2},
        {"chunk_size": 512, "overlap_ratio": 0.25},
        {"chunk_size": 512, "overlap_ratio": 0.3},
        {"chunk_size": 768, "overlap_ratio": 0.2},
        {"chunk_size": 1024, "overlap_ratio": 0.3},
    ]
    TOP_K = 10
    CSV_PATH = Path(__file__).parent.parent / "medium-english-50mb.csv"

    # Load subset of articles once
    articles: list[dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= subset:
                break
            if (row.get("text") or "").strip():
                articles.append({
                    "id": str(i),
                    "title": (row.get("title") or "").strip(),
                    "text":  (row.get("text")  or "").strip(),
                    "url":   (row.get("url")   or "").strip(),
                    "authors": (row.get("authors") or "").strip(),
                })

    results = []
    for cfg in CONFIGS:
        chunk_words   = round(cfg["chunk_size"] * 0.75)
        overlap_words = round(chunk_words * cfg["overlap_ratio"])
        index_name    = f"eval-cs{cfg['chunk_size']}-ov{int(cfg['overlap_ratio']*10)}"

        # Build chunks
        all_vectors: list[dict] = []
        for art in articles:
            for j, chunk in enumerate(chunk_text(art["text"], chunk_words, overlap_words)):
                all_vectors.append({
                    "id":     f"a{art['id']}_c{j}",
                    "text":   chunk,
                    "metadata": {
                        "article_id": art["id"], "title": art["title"],
                        "chunk_text": chunk,     "url":   art["url"],
                        "authors":    art["authors"],
                    },
                })

        print(f"\n  Config chunk_size={cfg['chunk_size']} overlap={cfg['overlap_ratio']} "
              f"→ {len(all_vectors)} chunks. Embedding…")

        # Embed in batches of 50
        embeddings: list[list[float]] = []
        texts = [v["text"] for v in all_vectors]
        for i in range(0, len(texts), 50):
            embeddings.extend(embed_batch(texts[i : i + 50]))
            time.sleep(0.2)

        vectors = [
            {"id": v["id"], "values": emb, "metadata": v["metadata"]}
            for v, emb in zip(all_vectors, embeddings)
        ]

        # Upload to temp index
        print(f"  Uploading to temp index '{index_name}'…")
        build_temp_index(index_name, vectors)

        # Evaluate
        scores_all, div_all, quality_all = [], [], []
        for q in TEST_QUESTIONS:
            answer, matches = ask(index_name, q, TOP_K)
            m = metrics(matches, TOP_K)
            scores_all.append(m["avg_score"])
            div_all.append(m["diversity"])
            quality_all.append(judge(q, answer))
            time.sleep(0.3)

        results.append({
            "chunk_size":   cfg["chunk_size"],
            "overlap":      cfg["overlap_ratio"],
            "n_chunks":     len(all_vectors),
            "avg_score":    round(sum(scores_all)   / len(scores_all),  4),
            "diversity":    round(sum(div_all)       / len(div_all),     3),
            "quality(1-5)": round(sum(quality_all)  / len(quality_all), 2),
        })

        # Clean up temp index to save Pinecone quota
        pc.delete_index(index_name)
        print(f"  Done. Temp index deleted.")

    title = f"Chunk grid results (subset={subset} articles, top_k={TOP_K})"
    print_table(results, title)
    return results, title


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["topk", "chunks", "all"], default="topk",
        help="topk=sweep top_k only (free); chunks=re-embed subset; all=both",
    )
    parser.add_argument("--subset", type=int, default=200,
                        help="Articles to use for chunk grid (default 200)")
    args = parser.parse_args()

    sections = []

    if args.mode in ("topk", "all"):
        eval_topk()

    if args.mode in ("chunks", "all"):
        rows, title = eval_chunks(args.subset)
        sections.append(("chunks", rows, title))

    if sections:
        out = Path(__file__).parent.parent / "evaluation_results.txt"
        save_report(sections, out)


if __name__ == "__main__":
    main()
