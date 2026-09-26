from defect_triage_agent.vector_store import VectorStore


def test_vector_store_retrieves_related_issue_with_tfidf() -> None:
    issues = [
        {
            "number": 1,
            "title": "Crash opening notebook with many outputs",
            "body": "The editor freezes when rendering a large notebook.",
            "category": "bug",
            "labels": ["bug"],
            "url": "https://example.com/1",
        },
        {
            "number": 2,
            "title": "Add a new color theme",
            "body": "It would be nice to have another theme.",
            "category": "enhancement",
            "labels": ["enhancement"],
            "url": "https://example.com/2",
        },
    ]
    store = VectorStore.build(issues, backend_name="tfidf")
    results = store.search("Notebook crashes with large output", top_k=1)
    assert results[0].issue["number"] == 1

