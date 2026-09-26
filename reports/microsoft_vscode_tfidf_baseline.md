# Baseline: microsoft/vscode With TF-IDF Retrieval

Run date: 2026-09-27

This is a smoke-test baseline using public GitHub Issues from `microsoft/vscode`.

## Dataset

- Raw public issues fetched: 447
- Labeled issues after category normalization: 95
- Reference issues: 75
- Held-out test issues: 20
- Labels present: `bug`, `enhancement`, `duplicate`

## Configuration

- Embedding backend: `tfidf`
- Retrieval: local NumPy cosine similarity
- Top-k examples: 5
- LLM: disabled

## Result

- Correct: 15 / 20
- Accuracy: 0.75

## Confusion Matrix

| Human label | Predicted bug | Predicted enhancement | Predicted duplicate |
| --- | ---: | ---: | ---: |
| bug | 13 | 0 | 0 |
| enhancement | 3 | 2 | 0 |
| duplicate | 2 | 0 | 0 |

## Notes

This baseline is intentionally modest. TF-IDF is included for fast local testing, not as the final semantic retrieval claim. The stronger project path is to rebuild the same index with `sentence-transformers` or OpenAI embeddings and rerun the same held-out evaluation.

