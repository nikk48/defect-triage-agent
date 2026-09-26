#!/usr/bin/env python
from __future__ import annotations

import argparse

from defect_triage_agent.io import read_jsonl, slugify_repo
from defect_triage_agent.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector index from reference issues.")
    parser.add_argument("--repo", required=True, help="Repository in owner/name form.")
    parser.add_argument(
        "--backend",
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "tfidf"],
    )
    parser.add_argument("--infile", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    slug = slugify_repo(args.repo)
    infile = args.infile or f"data/processed/{slug}_reference.jsonl"
    out = args.out or f"data/indexes/{slug}_{args.backend}"
    issues = list(read_jsonl(infile))
    store = VectorStore.build(issues, backend_name=args.backend)
    store.save(out)
    print(f"Built {args.backend} index with {len(issues)} issues at {out}")


if __name__ == "__main__":
    main()

