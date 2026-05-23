"""Build a raw-docstring RAG index over omicverse public callables.

This intentionally walks the package-level namespaces and extracts only
``inspect.signature`` + ``inspect.getdoc`` — it does NOT consult any of the
Beacon-side metadata: ``aliases``, ``requires`` / ``produces``,
``prerequisites``, ``auto_fix``, ``skill_*`` — i.e. the structured
library-side contract this ablation is comparing against.

Output: pickle at ``data/doc_rag_index/index.pkl`` containing:
  - chunks: list of {'name', 'text'} dicts (one per public callable)
  - embeddings: float32 (n_chunks, 384) numpy array
  - model_name: 'sentence-transformers/all-MiniLM-L6-v2'

The companion ``scripts/doc_lookup.py`` loads this index and exposes a
``doc_lookup(query)`` callable that the agent can invoke from bash.
"""
from __future__ import annotations

import inspect
import pickle
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


# Public namespaces to walk. We deliberately skip ``ov.utils.*`` because
# the registry/skill scanners live there — including their docstrings
# would leak Beacon hints into the RAG corpus.
SUBMODULES = [
    "omicverse.bulk",
    "omicverse.single",
    "omicverse.space",
    "omicverse.micro",
    "omicverse.pp",
    "omicverse.pl",
    "omicverse.popv",
    "omicverse.bulk2single",
    "omicverse.alignment",
    "omicverse.external.scllm",
    "omicverse.external.PyWGCNA",
    "omicverse.external.single",
]


def _iter_public_callables(mod):
    seen = set()
    for name in dir(mod):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        if not callable(obj):
            continue
        # Avoid duplicates from re-exports.
        try:
            qual = f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', name)}"
        except Exception:
            qual = name
        if qual in seen:
            continue
        seen.add(qual)
        yield name, obj


def _chunk_for(obj, full_name: str) -> str | None:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"
    doc = inspect.getdoc(obj) or ""
    if len(doc.strip()) < 30:
        # Skip empty / one-liner stubs — RAG over those is noise.
        return None
    # Cap each chunk so a single long docstring doesn't dominate the corpus.
    if len(doc) > 4000:
        doc = doc[:4000].rsplit("\n\n", 1)[0] + "\n[...]"
    return f"### {full_name}{sig}\n\n{doc}"


def main():
    import importlib
    chunks: list[dict] = []
    seen_names: set[str] = set()

    for submod_name in SUBMODULES:
        try:
            mod = importlib.import_module(submod_name)
        except Exception as exc:
            print(f"[warn] cannot import {submod_name}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        for name, obj in _iter_public_callables(mod):
            full_name = f"{submod_name}.{name}"
            if full_name in seen_names:
                continue
            seen_names.add(full_name)
            text = _chunk_for(obj, full_name)
            if text is None:
                continue
            chunks.append({"name": full_name, "text": text})
            # also try class methods for top-level classes
            if inspect.isclass(obj):
                for mname in dir(obj):
                    if mname.startswith("_"):
                        continue
                    try:
                        m = getattr(obj, mname)
                    except Exception:
                        continue
                    if not callable(m):
                        continue
                    sub_full = f"{full_name}.{mname}"
                    if sub_full in seen_names:
                        continue
                    seen_names.add(sub_full)
                    sub_text = _chunk_for(m, sub_full)
                    if sub_text is None:
                        continue
                    chunks.append({"name": sub_full, "text": sub_text})

    print(f"[corpus] collected {len(chunks)} chunks across {len(SUBMODULES)} submodules")

    # Embed
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    print(f"[embed] encoding {len(texts)} chunks with {model_name}...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    print(f"[embed] shape={embeddings.shape}  dtype={embeddings.dtype}")

    out_dir = Path(__file__).resolve().parents[1] / "data" / "doc_rag_index"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "embeddings": embeddings,
            "model_name": model_name,
        }, f)
    print(f"[done] wrote {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
