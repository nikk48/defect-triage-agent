from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass

from defect_triage_agent.labels import CANONICAL_LABELS
from defect_triage_agent.vector_store import SearchResult


@dataclass(frozen=True)
class TriageSuggestion:
    category: str
    root_cause_hypothesis: str
    confidence: float
    rationale: str
    citations: list[dict]
    used_llm: bool

    def to_dict(self) -> dict:
        return asdict(self)


def triage_issue(
    title: str,
    body: str,
    retrieved: list[SearchResult],
    *,
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
) -> TriageSuggestion:
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            return _llm_triage(title, body, retrieved, model=model)
        except Exception as exc:
            fallback = _retrieval_triage(title, body, retrieved)
            return TriageSuggestion(
                category=fallback.category,
                root_cause_hypothesis=fallback.root_cause_hypothesis,
                confidence=min(fallback.confidence, 0.5),
                rationale=f"LLM call failed, so this used retrieval fallback. Error: {exc}",
                citations=fallback.citations,
                used_llm=False,
            )
    return _retrieval_triage(title, body, retrieved)


def _retrieval_triage(title: str, body: str, retrieved: list[SearchResult]) -> TriageSuggestion:
    weighted = Counter()
    citations = []
    for result in retrieved:
        issue = result.issue
        category = issue.get("category")
        if category:
            weighted[category] += max(result.score, 0.0)
        citations.append(_citation(result))

    if weighted:
        category, top_weight = weighted.most_common(1)[0]
        total = sum(weighted.values()) or 1.0
        margin = top_weight / total
        confidence = round(min(0.95, max(0.25, margin)), 2)
    else:
        category = "bug"
        confidence = 0.25

    top = retrieved[0].issue if retrieved else {}
    root_cause = (
        "Likely related to behavior seen in the nearest historical issue: "
        f"{top.get('title', 'no close example available')}"
    )
    rationale = (
        "Category is based on similarity-weighted voting across retrieved public issues. "
        "Use the LLM mode for a richer hypothesis when OPENAI_API_KEY is configured."
    )
    return TriageSuggestion(
        category=category,
        root_cause_hypothesis=root_cause,
        confidence=confidence,
        rationale=rationale,
        citations=citations,
        used_llm=False,
    )


def _llm_triage(
    title: str,
    body: str,
    retrieved: list[SearchResult],
    *,
    model: str,
) -> TriageSuggestion:
    from openai import OpenAI

    client = OpenAI()
    examples = [
        {
            "number": result.issue.get("number"),
            "title": result.issue.get("title"),
            "category": result.issue.get("category"),
            "labels": result.issue.get("labels", []),
            "url": result.issue.get("url"),
            "similarity": round(result.score, 3),
            "body_excerpt": (result.issue.get("body") or "")[:900],
        }
        for result in retrieved
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a defect-triage assistant. Ground your answer only in the new issue "
                "and the retrieved public GitHub Issues. Return valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "allowed_categories": list(CANONICAL_LABELS),
                    "new_issue": {"title": title, "body": body},
                    "retrieved_examples": examples,
                    "required_json_schema": {
                        "category": "one allowed category",
                        "root_cause_hypothesis": "short hypothesis grounded in examples",
                        "confidence": "number from 0 to 1",
                        "rationale": "brief explanation citing example issue numbers",
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    category = payload.get("category", "bug")
    if category not in CANONICAL_LABELS:
        category = "bug"
    confidence = float(payload.get("confidence", 0.5))
    confidence = round(min(1.0, max(0.0, confidence)), 2)
    return TriageSuggestion(
        category=category,
        root_cause_hypothesis=payload.get("root_cause_hypothesis", ""),
        confidence=confidence,
        rationale=payload.get("rationale", ""),
        citations=[_citation(result) for result in retrieved],
        used_llm=True,
    )


def _citation(result: SearchResult) -> dict:
    issue = result.issue
    return {
        "number": issue.get("number"),
        "title": issue.get("title"),
        "category": issue.get("category"),
        "url": issue.get("url"),
        "similarity": round(result.score, 3),
    }

