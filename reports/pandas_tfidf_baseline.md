# Baseline: pandas-dev/pandas With TF-IDF Retrieval

Run date: 2026-09-27

This is a smoke-test baseline using public GitHub Issues from `pandas-dev/pandas`.

## Dataset

- Raw public issues fetched: 118
- Labeled issues after category normalization: 78
- Reference issues: 62
- Held-out test issues: 16
- Labels present: `bug`, `enhancement`, `documentation`, `duplicate`

## Configuration

- Embedding backend: `tfidf`
- Retrieval: local NumPy cosine similarity
- Top-k examples: 5
- LLM: disabled

## Result

- Correct: 15 / 16
- Accuracy: 0.9375

## Confusion Matrix

| Human label | Predicted bug | Predicted enhancement | Predicted documentation |
| --- | ---: | ---: | ---: |
| bug | 10 | 1 | 0 |
| enhancement | 0 | 2 | 0 |
| documentation | 0 | 0 | 3 |

## Notes

This result comes from a small held-out sample and should be presented as a demo baseline, not as a broad claim about all pandas issues. It is useful because it proves the same pipeline can be pointed at another public repository without changing the core agent code.

