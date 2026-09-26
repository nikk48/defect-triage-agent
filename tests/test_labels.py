from defect_triage_agent.labels import add_primary_labels, primary_label, stratified_split


def test_primary_label_prioritizes_duplicate_over_bug() -> None:
    assert primary_label(["bug", "duplicate"]) == "duplicate"


def test_primary_label_maps_common_aliases() -> None:
    assert primary_label(["feature-request"]) == "enhancement"
    assert primary_label(["docs"]) == "documentation"
    assert primary_label(["regression"]) == "bug"


def test_add_primary_labels_filters_unlabeled_rows() -> None:
    rows = [
        {"number": 1, "labels": ["bug"]},
        {"number": 2, "labels": ["needs author feedback"]},
    ]
    labeled = add_primary_labels(rows)
    assert len(labeled) == 1
    assert labeled[0]["category"] == "bug"


def test_stratified_split_keeps_each_label_in_reference() -> None:
    rows = [{"category": "bug", "number": i} for i in range(10)]
    reference, test = stratified_split(rows, test_size=0.3, seed=1)
    assert reference
    assert test
    assert {row["category"] for row in reference} == {"bug"}

