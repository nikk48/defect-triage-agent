from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable

import requests


GITHUB_API_ROOT = "https://api.github.com"


@dataclass(frozen=True)
class IssueRecord:
    id: int
    number: int
    title: str
    body: str
    labels: list[str]
    url: str
    state: str
    created_at: str
    closed_at: str | None

    @property
    def text(self) -> str:
        body = self.body or ""
        return f"{self.title}\n\n{body}".strip()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "number": self.number,
            "title": self.title,
            "body": self.body or "",
            "labels": self.labels,
            "url": self.url,
            "state": self.state,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "text": self.text,
        }


def _headers(token: str | None = None) -> dict[str, str]:
    token = token or os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "defect-triage-agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_repo_issues(
    repo: str,
    *,
    state: str = "all",
    max_pages: int | None = None,
    per_page: int = 100,
    token: str | None = None,
    sleep_seconds: float = 0.2,
) -> Iterable[IssueRecord]:
    """Fetch public GitHub Issues, excluding pull requests returned by the Issues API."""
    if "/" not in repo:
        raise ValueError("repo must look like 'owner/name'")

    session = requests.Session()
    session.headers.update(_headers(token))

    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break

        response = session.get(
            f"{GITHUB_API_ROOT}/repos/{repo}/issues",
            params={
                "state": state,
                "per_page": per_page,
                "page": page,
                "sort": "created",
                "direction": "desc",
            },
            timeout=30,
        )

        if response.status_code == 403 and "rate limit" in response.text.lower():
            raise RuntimeError(
                "GitHub API rate limit hit. Set GITHUB_TOKEN for a higher public API limit."
            )
        if response.status_code == 422 and page > 1:
            break
        response.raise_for_status()

        payload = response.json()
        if not payload:
            break

        for raw in payload:
            if "pull_request" in raw:
                continue
            labels = [label.get("name", "") for label in raw.get("labels", []) if label.get("name")]
            yield IssueRecord(
                id=raw["id"],
                number=raw["number"],
                title=raw.get("title") or "",
                body=raw.get("body") or "",
                labels=labels,
                url=raw.get("html_url") or "",
                state=raw.get("state") or "",
                created_at=raw.get("created_at") or "",
                closed_at=raw.get("closed_at"),
            )

        page += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)
