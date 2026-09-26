from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from defect_triage_agent.embeddings import EmbeddingBackend, create_backend, load_backend
from defect_triage_agent.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class SearchResult:
    issue: dict
    score: float


class VectorStore:
    def __init__(self, backend: EmbeddingBackend, issues: list[dict], vectors: np.ndarray) -> None:
        self.backend = backend
        self.issues = issues
        self.vectors = _l2_normalize(vectors)

    @classmethod
    def build(cls, issues: list[dict], backend_name: str) -> "VectorStore":
        if not issues:
            raise ValueError("Cannot build a vector store with zero issues.")
        backend = create_backend(backend_name)
        texts = [issue_text(issue) for issue in issues]
        vectors = backend.fit_transform(texts)
        return cls(backend, issues, vectors)

    def search(self, title: str, body: str = "", *, top_k: int = 5) -> list[SearchResult]:
        text = f"{title}\n\n{body}".strip()
        if not text:
            raise ValueError("Search text cannot be empty.")
        query = _l2_normalize(self.backend.transform([text]))[0]
        scores = self.vectors @ query
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(issue=self.issues[index], score=float(scores[index]))
            for index in top_indices
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        write_jsonl(path / "issues.jsonl", self.issues)
        metadata = {
            "backend": self.backend.name,
            "count": len(self.issues),
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.backend.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        backend = load_backend(metadata["backend"], path)
        issues = list(read_jsonl(path / "issues.jsonl"))
        vectors = np.load(path / "vectors.npy")
        return cls(backend, issues, vectors)


def issue_text(issue: dict) -> str:
    title = issue.get("title", "")
    body = issue.get("body") or issue.get("text", "")
    return f"{title}\n\n{body}".strip()


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

