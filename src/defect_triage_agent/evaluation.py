from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from defect_triage_agent.io import write_jsonl
from defect_triage_agent.triage_agent import triage_issue
from defect_triage_agent.vector_store import VectorStore


def evaluate(
    store: VectorStore,
    test_rows: Iterable[dict],
    *,
    top_k: int = 5,
    use_llm: bool = False,
    max_test: int | None = None,
    report_path: str | Path | None = None,
) -> dict:
    rows = list(test_rows)
    if max_test is not None:
        rows = rows[:max_test]
    if not rows:
        raise ValueError("No test rows to evaluate.")

    predictions = []
    correct = 0
    confusion: dict[str, Counter] = defaultdict(Counter)

    for row in tqdm(rows, desc="Evaluating"):
        retrieved = store.search(row.get("title", ""), row.get("body", ""), top_k=top_k)
        suggestion = triage_issue(
            row.get("title", ""),
            row.get("body", ""),
            retrieved,
            use_llm=use_llm,
        )
        expected = row["category"]
        predicted = suggestion.category
        is_correct = expected == predicted
        correct += int(is_correct)
        confusion[expected][predicted] += 1
        predictions.append(
            {
                "number": row.get("number"),
                "title": row.get("title"),
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "confidence": suggestion.confidence,
                "used_llm": suggestion.used_llm,
                "citations": suggestion.citations,
                "url": row.get("url"),
            }
        )

    accuracy = correct / len(rows)
    summary = {
        "n_test": len(rows),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "top_k": top_k,
        "used_llm": use_llm,
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
        "predictions": predictions,
    }
    if report_path:
        write_jsonl(report_path, predictions)
    return summary

