#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from defect_triage_agent.evaluation import evaluate
from defect_triage_agent.io import read_jsonl, slugify_repo
from defect_triage_agent.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out label agreement.")
    parser.add_argument("--repo", required=True, help="Repository in owner/name form.")
    parser.add_argument(
        "--backend",
        default="sentence-transformers",
        choices=["sentence-transformers", "openai", "tfidf"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic retrieval-only triage.")
    parser.add_argument("--index", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    slug = slugify_repo(args.repo)
    index_path = args.index or f"data/indexes/{slug}_{args.backend}"
    test_path = args.test or f"data/processed/{slug}_test.jsonl"
    report_path = args.report or f"eval_reports/{slug}_{args.backend}_predictions.jsonl"

    store = VectorStore.load(index_path)
    summary = evaluate(
        store,
        read_jsonl(test_path),
        top_k=args.top_k,
        use_llm=not args.no_llm,
        max_test=args.max_test,
        report_path=report_path,
    )
    printable = {key: value for key, value in summary.items() if key != "predictions"}
    print(json.dumps(printable, indent=2))
    print(f"Prediction-level report: {report_path}")


if __name__ == "__main__":
    main()

