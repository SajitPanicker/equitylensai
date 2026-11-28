# Offline Ragas evaluation

Equity Lens evaluation is intentionally outside the Streamlit request path. The
module in [`evaluation/ragas_evaluation.py`](../evaluation/ragas_evaluation.py)
reuses the existing RAG subgraph and injected retrieval functions, but it is
only run explicitly.

## Dataset

[`evaluation/benchmark_dataset.json`](../evaluation/benchmark_dataset.json)
contains the initial 30-question benchmark. Each record includes the question,
company, ticker, period, document type, verification status, ground truth, and
official reference sources.

The initial records are `pending_verification` and intentionally contain no
financial figures or ground-truth answers. Before a record can be scored:

1. Verify the answer manually against an official NSE/BSE filing, annual report,
   or company investor-relations document.
2. Add the exact verified answer to `ground_truth`.
3. Add one or more official URLs and page/slide references to `reference_sources`.
4. Change `verification_status` to `verified`.

The loader rejects verified records with missing ground truth or reference
sources. Pending records remain in pipeline reports but are excluded from Ragas
scoring.

## Retrieval comparison

The evaluator runs the same benchmark records, filters, retrieval settings, and
answer-generation settings twice:

- `baseline`: existing ChromaDB retrieval, with no reranker callback.
- `enhanced`: existing ChromaDB retrieval followed by the existing Cohere
  reranker callback.

The production retrieval and answer functions are injected into the existing
`build_rag_graph` implementation. No retrieval logic is duplicated, and no
production setting is changed by baseline evaluation.

## Metrics

Ragas `0.4.0` is used with these metrics:

- `Faithfulness`: whether the answer is supported by retrieved context.
- `ResponseRelevancy` (reported as `answer_relevancy`): whether the answer
  addresses the question.
- `LLMContextPrecisionWithoutReference`: whether useful contexts are ranked
  earlier.
- `LLMContextRecall`: whether the retrieved context supports the verified
  reference answer.

Scores are retrieval-quality and answer-grounding indicators, not proof that a
source filing was extracted correctly. Reports preserve complete retrieved
document metadata, source URLs, page/slide locations, and `source_references`.
Rows with incomplete source metadata are flagged for manual review.

## Running the evaluation

Install the pinned evaluator dependency and the existing project dependencies:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

After setting the existing OpenAI and optional Cohere environment variables,
run:

```powershell
& .\.venv\Scripts\python.exe -m evaluation.ragas_evaluation
```

Optional paths are supported:

```powershell
& .\.venv\Scripts\python.exe -m evaluation.ragas_evaluation `
  --benchmark evaluation/benchmark_dataset.json `
  --output evaluation/results
```

The command writes:

- `baseline_results.json` and `baseline_results.csv`
- `enhanced_results.json` and `enhanced_results.csv`

Evaluation calls use the configured answer model and Ragas metric LLM/embedding
adapters, so running a verified benchmark can incur OpenAI charges. Cohere
costs apply only to enhanced runs when a Cohere key is configured. Automated
tests mock all paid model, embedding, Cohere, and Ragas evaluator calls.

## Limitations

- The initial records require manual official-source verification before they
  produce scores.
- Ragas cannot independently establish that a filing was extracted correctly.
- A high faithfulness score can still reflect a faithfully repeated extraction
  error.
- Baseline and enhanced runs are comparable only when executed against the same
  indexed ChromaDB snapshot and configuration.
