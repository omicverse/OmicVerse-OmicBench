"""Vanilla doc-RAG lookup for the ablation arm.

Loads the embedding index built by ``build_doc_rag_index.py`` and prints
the top-K matching docstring chunks for a given query. The chunks are
the raw output of ``inspect.signature`` + ``inspect.getdoc`` over
omicverse's public callables — no Beacon-side metadata (aliases,
requires/produces, prerequisites, auto_fix, skill recipes) is included.

Usage from the agent's bash::

    python <OVBENCH_ROOT>/scripts/doc_lookup.py "cell type annotation"
    python <OVBENCH_ROOT>/scripts/doc_lookup.py --k 4 "single-cell GRN"
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


_INDEX_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "doc_rag_index" / "index.pkl"
)


def _load_index():
    with open(_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def doc_lookup(query: str, k: int = 8, max_chars_per_chunk: int = 1500) -> str:
    """Return the top-K omicverse docstring chunks most similar to ``query``.

    Pure embedding-RAG: ranks raw docstrings by cosine similarity.
    """
    idx = _load_index()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(idx["model_name"])
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = (idx["embeddings"] @ q.T).ravel()
    top = np.argsort(-sims)[: int(k)]
    out = []
    for rank, i in enumerate(top, 1):
        chunk = idx["chunks"][i]["text"]
        if len(chunk) > max_chars_per_chunk:
            chunk = chunk[:max_chars_per_chunk] + "\n[...truncated]"
        out.append(f"--- match {rank}/{k}  similarity={float(sims[i]):.3f} ---\n{chunk}")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="natural-language query")
    ap.add_argument("--k", type=int, default=8, help="top-K (default 8)")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="cap each returned chunk (default 1500)",
    )
    args = ap.parse_args()
    print(doc_lookup(args.query, k=args.k, max_chars_per_chunk=args.max_chars))


if __name__ == "__main__":
    main()
