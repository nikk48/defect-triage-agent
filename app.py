from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from defect_triage_agent.io import slugify_repo
from defect_triage_agent.triage_agent import triage_issue
from defect_triage_agent.vector_store import VectorStore


st.set_page_config(page_title="Defect Triage Agent", layout="wide")

EXAMPLE_ISSUES = {
    "django/django": {
        "title": "Migration fails when renaming a model with a foreign key",
        "body": (
            "After renaming a model that is referenced by a ForeignKey, running migrations fails "
            "with an invalid table reference. Expected behavior: Django should generate migrations "
            "that preserve the relationship. Actual behavior: migrate fails on PostgreSQL during "
            "the schema change."
        ),
    },
    "fastapi/fastapi": {
        "title": "Response model validation fails for optional nested fields",
        "body": (
            "An endpoint returning a Pydantic model with optional nested fields raises a validation "
            "error when the nested object is None. Expected behavior: FastAPI should serialize the "
            "response with the optional field as null. Actual behavior: the request returns a 500."
        ),
    },
    "matplotlib/matplotlib": {
        "title": "Legend overlaps plot after tight_layout with multiple subplots",
        "body": (
            "When creating a figure with multiple subplots and calling tight_layout, the legend "
            "overlaps the plotted data in the lower-right subplot. Expected behavior: layout should "
            "reserve enough space for the legend. Actual behavior: labels and legend entries are clipped."
        ),
    },
    "microsoft/vscode": {
        "title": "Editor freezes when opening a large notebook file",
        "body": (
            "After the latest update, opening a notebook with many output cells causes the editor "
            "window to freeze. After a few seconds, the app becomes unresponsive and sometimes crashes. "
            "Expected behavior: the notebook should open normally or progressively load outputs. "
            "Actual behavior: VS Code freezes and must be restarted."
        ),
    },
    "numpy/numpy": {
        "title": "np.linalg solve returns inconsistent result for float32 matrix",
        "body": (
            "Solving a small linear system with a float32 matrix returns a result that differs from "
            "the same operation using float64. Expected behavior: the float32 result should be close "
            "within tolerance. Actual behavior: one element is far outside the expected tolerance."
        ),
    },
    "pandas-dev/pandas": {
        "title": "DataFrame groupby returns incorrect totals after sorting by date",
        "body": (
            "I have a DataFrame with a date column, a category column, and a numeric value column. "
            "After sorting by date and then running groupby on the category column, the aggregated "
            "totals do not match the totals from the original unsorted DataFrame. Expected behavior: "
            "groupby should return the same totals regardless of row order."
        ),
    },
    "scikit-learn/scikit-learn": {
        "title": "RandomForestClassifier gives different predictions after reload",
        "body": (
            "After fitting a RandomForestClassifier and saving it with joblib, loading the model again "
            "produces different predictions on the same test data. Expected behavior: predictions should "
            "be identical after serialization. Actual behavior: a small subset of labels changes."
        ),
    },
    "streamlit/streamlit": {
        "title": "st.dataframe selection resets after rerun",
        "body": (
            "When selecting rows in st.dataframe and then interacting with another widget, the selected "
            "rows reset after the rerun. Expected behavior: the selection should remain in session state. "
            "Actual behavior: the selection becomes empty."
        ),
    },
}


@st.cache_resource(show_spinner=False)
def load_store(path: str) -> VectorStore:
    return VectorStore.load(path)


def available_indexes(repo: str) -> list[Path]:
    slug = slugify_repo(repo)
    return sorted(
        path
        for path in Path("data/indexes").glob(f"{slug}_*")
        if path.is_dir() and (path / "metadata.json").exists()
    )


def indexed_repos() -> list[str]:
    repos = set()
    for path in Path("data/indexes").glob("*_*"):
        if path.is_dir() and (path / "metadata.json").exists() and "__" in path.name:
            repo_slug = path.name.rsplit("_", 1)[0]
            repos.add(repo_slug.replace("__", "/"))
    return sorted(repos)


st.title("Defect Triage Agent")
st.caption("Public GitHub Issues only. Inspired by real defect triage work, rebuilt without confidential data.")

with st.sidebar:
    repo_options = indexed_repos()
    if repo_options:
        selected_repo = st.selectbox("Repository", repo_options + ["Custom repository"])
        if selected_repo == "Custom repository":
            repo = st.text_input("Repository path", value="owner/name")
        else:
            repo = selected_repo
    else:
        repo = st.text_input("Repository", value="microsoft/vscode")
    indexes = available_indexes(repo)
    if indexes:
        index_path = st.selectbox(
            "Local index",
            [str(path) for path in indexes],
            key="local_index_v1",
        )
    else:
        backend = st.selectbox("Index backend", ["sentence-transformers", "tfidf", "openai"])
        index_path = st.text_input(
            "Index path",
            value=f"data/indexes/{slugify_repo(repo)}_{backend}",
            key="manual_index_path_v1",
        )
    top_k = st.slider("Similar issues", min_value=3, max_value=10, value=5)
    use_llm = st.toggle("Use LLM if OPENAI_API_KEY is set", value=False)

example = EXAMPLE_ISSUES.get(repo, EXAMPLE_ISSUES["microsoft/vscode"])
repo_key = slugify_repo(repo)
title = st.text_input("New issue title", value=example["title"], key=f"title_{repo_key}")
body = st.text_area(
    "New issue body",
    value=example["body"],
    key=f"body_{repo_key}",
    height=180,
)

if st.button("Triage Issue", type="primary"):
    path = Path(index_path)
    if not path.exists():
        st.error(f"Index not found: {path}. Build it with scripts/build_index.py first.")
        st.stop()

    with st.spinner("Retrieving similar public issues..."):
        store = load_store(str(path))
        retrieved = store.search(title, body, top_k=top_k)
        suggestion = triage_issue(title, body, retrieved, use_llm=use_llm)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Suggested category", suggestion.category)
    metric_cols[1].metric("Confidence", f"{suggestion.confidence:.2f}")
    metric_cols[2].metric("Used LLM", "yes" if suggestion.used_llm else "no")

    st.subheader("Root-cause hypothesis")
    st.write(suggestion.root_cause_hypothesis)
    st.subheader("Rationale")
    st.write(suggestion.rationale)

    st.subheader("Retrieved examples")
    rows = [
        {
            "similarity": citation["similarity"],
            "category": citation["category"],
            "number": citation["number"],
            "title": citation["title"],
            "url": citation["url"],
        }
        for citation in suggestion.citations
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
