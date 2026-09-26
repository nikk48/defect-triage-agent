from __future__ import annotations

from collections import Counter, defaultdict
from random import Random
from typing import Iterable


CANONICAL_LABELS = ("bug", "duplicate", "enhancement", "documentation", "question")

LABEL_ALIASES = {
    "duplicate": ("duplicate", "dupe", "duplicated"),
    "bug": ("bug", "defect", "crash", "regression", "error", "fault"),
    "enhancement": ("enhancement", "feature", "feature-request", "improvement", "request"),
    "documentation": ("documentation", "docs", "doc", "readme"),
    "question": ("question", "support", "help wanted", "needs more info"),
}

LABEL_PRIORITY = ("duplicate", "bug", "enhancement", "documentation", "question")


def normalize_label(raw_label: str) -> str | None:
    label = raw_label.strip().lower()
    if not label:
        return None

    for canonical, aliases in LABEL_ALIASES.items():
        for alias in aliases:
            if label == alias or alias in label:
                return canonical
    return None


def primary_label(labels: Iterable[str]) -> str | None:
    normalized = {normalize_label(label) for label in labels}
    normalized.discard(None)
    for label in LABEL_PRIORITY:
        if label in normalized:
            return label
    return None


def add_primary_labels(rows: Iterable[dict]) -> list[dict]:
    labeled = []
    for row in rows:
        category = primary_label(row.get("labels", []))
        if category:
            item = dict(row)
            item["category"] = category
            labeled.append(item)
    return labeled


def label_counts(rows: Iterable[dict]) -> dict[str, int]:
    return dict(Counter(row["category"] for row in rows if row.get("category")))


def stratified_split(
    rows: list[dict],
    *,
    test_size: float = 0.2,
    seed: int = 42,
    min_per_label: int = 2,
) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["category"]].append(row)

    rng = Random(seed)
    reference: list[dict] = []
    test: list[dict] = []

    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        if len(shuffled) < min_per_label:
            reference.extend(shuffled)
            continue
        n_test = max(1, int(round(len(shuffled) * test_size)))
        n_test = min(n_test, len(shuffled) - 1)
        test.extend(shuffled[:n_test])
        reference.extend(shuffled[n_test:])

    rng.shuffle(reference)
    rng.shuffle(test)
    return reference, test

