#!/usr/bin/env python
from __future__ import annotations

import argparse

from defect_triage_agent.github_issues import fetch_repo_issues
from defect_triage_agent.io import slugify_repo, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public GitHub Issues.")
    parser.add_argument("--repo", required=True, help="Repository in owner/name form.")
    parser.add_argument("--max-pages", type=int, default=5, help="100 issues per page.")
    parser.add_argument("--state", default="all", choices=["open", "closed", "all"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or f"data/raw/{slugify_repo(args.repo)}_issues.jsonl"
    rows = (issue.to_dict() for issue in fetch_repo_issues(args.repo, state=args.state, max_pages=args.max_pages))
    count = write_jsonl(out, rows)
    print(f"Wrote {count} public issues to {out}")


if __name__ == "__main__":
    main()

