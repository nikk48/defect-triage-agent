#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from defect_triage_agent.triage_agent import triage_issue
from defect_triage_agent.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage one new issue against a built index.")
    parser.add_argument("--index", required=True, help="Path created by scripts/build_index.py.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--body", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    store = VectorStore.load(args.index)
    retrieved = store.search(args.title, args.body, top_k=args.top_k)
    suggestion = triage_issue(
        args.title,
        args.body,
        retrieved,
        use_llm=not args.no_llm,
    )
    print(json.dumps(suggestion.to_dict(), indent=2))


if __name__ == "__main__":
    main()

