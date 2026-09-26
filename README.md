# Defect Triage Agent

A retrieval-augmented defect-triage assistant built on public GitHub Issues.

Given a new bug report, the app retrieves similar historical issues from a public repository and suggests:

- a likely triage category
- a root-cause hypothesis
- a confidence score
- citations to similar past issues

## Honest Project Context

This project is inspired by real defect-triage work I did at HPE, where I manually reviewed large batches of defects. It is rebuilt from scratch on public GitHub Issues because HPE defect data is confidential.

This repository does not contain HPE data, customer logs, internal tickets, proprietary labels, or private issue history.

## Why This Project Exists

Manual defect triage is repetitive but high-impact. Engineers often need to answer the same first questions:

- Has this happened before?
- Is this likely a bug, enhancement, duplicate, docs issue, or support question?
- Which past defects look similar?
- What is the likely root cause?

This project turns that workflow into a small, explainable RAG system.

## Features

- Fetches public GitHub Issues through the GitHub REST API.
- Filters out pull requests returned by the Issues API.
- Normalizes labels into common categories:
  - `bug`
  - `duplicate`
  - `enhancement`
  - `documentation`
  - `question`
- Creates reference and held-out test splits.
- Builds a local retrieval index.
- Supports multiple retrieval backends:
  - TF-IDF baseline
  - Sentence Transformers
  - OpenAI embeddings
- Runs retrieval-grounded triage from the terminal or Streamlit.
- Optionally calls an OpenAI model for richer structured triage.
- Evaluates predictions against held-out human labels.

## Tech Stack

- Python
- Streamlit
- GitHub REST API
- scikit-learn
- NumPy
- sentence-transformers
- OpenAI SDK
- pytest

The implementation intentionally avoids a heavy orchestration framework. The goal is to make the pipeline easy to explain line by line in an interview.

## Architecture

```text
Public GitHub Issues
        |
        v
Fetch issues through GitHub REST API
        |
        v
Normalize labels and split data
        |
        v
Embed reference issues
        |
        v
Local vector/TF-IDF index
        |
        v
Retrieve top-k similar issues
        |
        v
Triage suggestion and citations
        |
        v
Held-out evaluation
```

## Project Structure

```text
defect-triage-agent/
  app.py
  scripts/
    fetch_issues.py
    prepare_dataset.py
    build_index.py
    evaluate.py
    triage_issue.py
  src/defect_triage_agent/
    github_issues.py
    labels.py
    embeddings.py
    vector_store.py
    triage_agent.py
    evaluation.py
    io.py
  reports/
  tests/
```

Generated issue data, local indexes, and evaluation prediction files are ignored by git.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional, but recommended for higher GitHub API rate limits:

```bash
export GITHUB_TOKEN="your_public_github_token"
```

Optional, for OpenAI embeddings or LLM triage:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## Build an Index for a Public Repo

Example using `microsoft/vscode`:

```bash
PYTHONPATH=src python scripts/fetch_issues.py --repo microsoft/vscode --max-pages 5
PYTHONPATH=src python scripts/prepare_dataset.py --repo microsoft/vscode
PYTHONPATH=src python scripts/build_index.py --repo microsoft/vscode --backend tfidf
```

Run evaluation:

```bash
PYTHONPATH=src python scripts/evaluate.py --repo microsoft/vscode --backend tfidf --no-llm
```

Run the Streamlit app:

```bash
PYTHONPATH=src streamlit run app.py
```

## Try a Single Issue from the Terminal

```bash
PYTHONPATH=src python scripts/triage_issue.py \
  --index data/indexes/microsoft__vscode_tfidf \
  --title "Editor freezes when opening a large notebook file" \
  --body "After the latest update, opening a notebook with many output cells causes the editor window to freeze and sometimes crash." \
  --no-llm
```

## Example Inputs

Use realistic bug reports with:

- short symptom-focused title
- expected behavior
- actual behavior
- environment or reproduction detail

Example for `pandas-dev/pandas`:

```text
Title:
DataFrame groupby returns incorrect totals after sorting by date

Body:
I have a DataFrame with a date column, a category column, and a numeric value column. After sorting by date and then running groupby on the category column, the aggregated totals do not match the totals from the original unsorted DataFrame.

Expected behavior: groupby should return the same totals regardless of row order.

Actual behavior: some categories have different totals after sorting.

This happens consistently on pandas 2.x with Python 3.11.
```

Example for `scikit-learn/scikit-learn`:

```text
Title:
RandomForestClassifier gives different predictions after reload

Body:
After fitting a RandomForestClassifier and saving it with joblib, loading the model again produces different predictions on the same test data.

Expected behavior: predictions should be identical after serialization.

Actual behavior: a small subset of labels changes after reload.
```

## Current Demo Baselines

These are small public-data smoke-test baselines using TF-IDF retrieval only. They are useful for demonstrating the evaluation loop, not for claiming production-level accuracy.

| Repository | Raw issues fetched | Labeled issues | Held-out test size | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `microsoft/vscode` | 447 | 95 | 20 | 75.00% |
| `pandas-dev/pandas` | 118 | 78 | 16 | 93.75% |
| `scikit-learn/scikit-learn` | 112 | 65 | 13 | 76.92% |
| `numpy/numpy` | 65 | 46 | 9 | 77.78% |
| `matplotlib/matplotlib` | 116 | 32 | 7 | 42.86% |

Detailed baseline notes are in `reports/`.

## Retrieval Backends

TF-IDF baseline:

```bash
PYTHONPATH=src python scripts/build_index.py --repo microsoft/vscode --backend tfidf
```

Sentence Transformers:

```bash
PYTHONPATH=src python scripts/build_index.py --repo microsoft/vscode --backend sentence-transformers
```

OpenAI embeddings:

```bash
export OPENAI_API_KEY="your_openai_api_key"
PYTHONPATH=src python scripts/build_index.py --repo microsoft/vscode --backend openai
```

## Optional LLM Triage

By default, the app can run without an LLM by using retrieval-weighted label voting.

To enable richer root-cause and rationale generation, set:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Then run evaluation without `--no-llm`:

```bash
PYTHONPATH=src python scripts/evaluate.py --repo microsoft/vscode --backend tfidf
```

## Add More Repositories

Replace `owner/name` with any public GitHub repository:

```bash
PYTHONPATH=src python scripts/fetch_issues.py --repo owner/name --max-pages 5
PYTHONPATH=src python scripts/prepare_dataset.py --repo owner/name
PYTHONPATH=src python scripts/build_index.py --repo owner/name --backend tfidf
PYTHONPATH=src python scripts/evaluate.py --repo owner/name --backend tfidf --no-llm
```

The Streamlit app auto-detects built indexes under `data/indexes/` and adds them to the repository dropdown.

## Tests

```bash
python -m pytest -q
PYTHONPATH=src python -m compileall -q app.py src scripts
```

## What This Is Not

- Not trained on HPE data.
- Not connected to HPE systems.
- Not a production incident-management system.
- Not a replacement for engineer judgment.

It is a transparent, public-data RAG prototype for demonstrating API integration, retrieval, structured prompting, evaluation, and a usable Streamlit demo.

