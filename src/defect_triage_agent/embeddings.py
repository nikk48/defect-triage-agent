from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingBackend(ABC):
    name: str

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path: Path) -> "EmbeddingBackend":
        raise NotImplementedError


class TfidfEmbeddingBackend(EmbeddingBackend):
    name = "tfidf"

    def __init__(self, vectorizer: TfidfVectorizer | None = None) -> None:
        self.vectorizer = vectorizer or TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        return matrix.astype(np.float32).toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts)
        return matrix.astype(np.float32).toarray()

    def save(self, path: Path) -> None:
        super().save(path)
        joblib.dump(self.vectorizer, path / "tfidf_vectorizer.joblib")

    @classmethod
    def load(cls, path: Path) -> "TfidfEmbeddingBackend":
        return cls(joblib.load(path / "tfidf_vectorizer.joblib"))


class SentenceTransformerBackend(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 25,
        )
        return np.asarray(vectors, dtype=np.float32)

    def save(self, path: Path) -> None:
        super().save(path)
        (path / "sentence_transformer_model.txt").write_text(self.model_name, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SentenceTransformerBackend":
        model_name = (path / "sentence_transformer_model.txt").read_text(encoding="utf-8").strip()
        return cls(model_name=model_name)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI embedding backend.")
        self.model = model
        self.client = OpenAI()

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        batch_size = 96
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return np.asarray(vectors, dtype=np.float32)

    def save(self, path: Path) -> None:
        super().save(path)
        (path / "openai_embedding_model.txt").write_text(self.model, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OpenAIEmbeddingBackend":
        model = (path / "openai_embedding_model.txt").read_text(encoding="utf-8").strip()
        return cls(model=model)


def create_backend(name: str) -> EmbeddingBackend:
    normalized = name.strip().lower()
    if normalized == "tfidf":
        return TfidfEmbeddingBackend()
    if normalized in {"sentence-transformers", "sentence_transformers", "sbert"}:
        return SentenceTransformerBackend()
    if normalized == "openai":
        return OpenAIEmbeddingBackend()
    raise ValueError(f"Unknown embedding backend: {name}")


def load_backend(name: str, path: Path) -> EmbeddingBackend:
    normalized = name.strip().lower()
    if normalized == "tfidf":
        return TfidfEmbeddingBackend.load(path)
    if normalized in {"sentence-transformers", "sentence_transformers", "sbert"}:
        return SentenceTransformerBackend.load(path)
    if normalized == "openai":
        return OpenAIEmbeddingBackend.load(path)
    raise ValueError(f"Unknown embedding backend: {name}")
