# Additional TF-IDF Baselines

Run date: 2026-09-27

These are small demo baselines from public GitHub Issues. They show that the same pipeline can be pointed at more than one public repository without changing the core agent code.

## scikit-learn/scikit-learn

- Raw public issues fetched: 112
- Labeled issues after normalization: 65
- Reference issues: 52
- Held-out test issues: 13
- Correct: 10 / 13
- Accuracy: 0.7692

## numpy/numpy

- Raw public issues fetched: 65
- Labeled issues after normalization: 46
- Reference issues: 37
- Held-out test issues: 9
- Correct: 7 / 9
- Accuracy: 0.7778

## matplotlib/matplotlib

- Raw public issues fetched: 116
- Labeled issues after normalization: 32
- Reference issues: 25
- Held-out test issues: 7
- Correct: 3 / 7
- Accuracy: 0.4286

## Not Yet Fetched

`streamlit/streamlit`, `fastapi/fastapi`, and `django/django` were requested, but GitHub's unauthenticated API rate limit was reached before those fetches completed. They can be added by rerunning the same fetch/prepare/build commands after the rate limit resets or after setting `GITHUB_TOKEN`.

