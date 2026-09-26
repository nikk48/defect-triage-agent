#!/usr/bin/env python
from __future__ import annotations

import argparse

from defect_triage_agent.io import read_jsonl, slugify_repo, write_jsonl
from defect_triage_agent.labels import add_primary_labels, label_counts, stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create labeled reference/test issue splits.")
    parser.add_argument("--repo", required=True, help="Repository in owner/name form.")
    parser.add_argument("--infile", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    slug = slugify_repo(args.repo)
    infile = args.infile or f"data/raw/{slug}_issues.jsonl"
    rows = add_primary_labels(read_jsonl(infile))
    reference, test = stratified_split(rows, test_size=args.test_size, seed=args.seed)

    reference_path = f"data/processed/{slug}_reference.jsonl"
    test_path = f"data/processed/{slug}_test.jsonl"
    write_jsonl(reference_path, reference)
    write_jsonl(test_path, test)

    print(f"Labeled issues: {len(rows)}")
    print(f"Label counts: {label_counts(rows)}")
    print(f"Reference: {len(reference)} -> {reference_path}")
    print(f"Test: {len(test)} -> {test_path}")


if __name__ == "__main__":
    main()

