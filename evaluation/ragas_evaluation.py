"""Offline Ragas evaluation for the Equity Lens retrieval pipeline.

This module is intentionally separate from the Streamlit and parent graph paths.
It reuses the existing RAG subgraph and injected retrieval/reranking functions,
but is only executed when called explicitly by an evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

REQUIRED_RECORD_FIELDS = {
    "question",
    "ground_truth",
    "company",
    "ticker",
    "fiscal_year",
    "quarter",
    "document_type",
    "reference_sources",
}
VERIFIED_STATUS = "verified"
PENDING_STATUS = "pending_verification"
REPORT_FIELDS = (
    "question",
    "company",
    "ticker",
    "fiscal_year",
    "quarter",
    "document_type",
    "verification_status",
    "retrieval_mode",
    "answer",
    "ground_truth",
    "retrieved_contexts",
    "retrieved_documents",
    "sources",
    "source_metadata_complete",
    "scoring_status",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "evaluation_error",
)


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark record cannot be evaluated safely."""


def validate_benchmark_record(record: dict[str, Any], index: int = 0) -> None:
    """Validate one benchmark record without inventing missing financial facts."""

    if not isinstance(record, dict):
        raise BenchmarkValidationError(f"Record {index} must be an object.")
    missing = sorted(REQUIRED_RECORD_FIELDS - record.keys())
    if missing:
        raise BenchmarkValidationError(
            f"Record {index} is missing required fields: {', '.join(missing)}"
        )
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise BenchmarkValidationError(f"Record {index} must have a non-empty question.")
    status = record.get("verification_status", PENDING_STATUS)
    if status not in {VERIFIED_STATUS, PENDING_STATUS}:
        raise BenchmarkValidationError(
            f"Record {index} has unsupported verification_status: {status!r}"
        )
    if not isinstance(record.get("reference_sources"), list):
        raise BenchmarkValidationError(f"Record {index} reference_sources must be a list.")
    if status == VERIFIED_STATUS:
        if not isinstance(record.get("ground_truth"), str) or not record["ground_truth"].strip():
            raise BenchmarkValidationError(
                f"Record {index} marked verified must contain ground_truth."
            )
        if not record["reference_sources"]:
            raise BenchmarkValidationError(
                f"Record {index} marked verified must contain reference_sources."
            )
    elif record.get("ground_truth") not in (None, ""):
        raise BenchmarkValidationError(
            f"Record {index} pending verification must not contain an unverified ground_truth."
        )


def load_benchmark_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the JSON benchmark dataset."""

    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise BenchmarkValidationError("Benchmark dataset must contain a JSON array.")
    for index, record in enumerate(records):
        validate_benchmark_record(record, index)
    return records


def scoreable_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only manually verified records eligible for Ragas scoring."""

    return [record for record in records if record.get("verification_status") == VERIFIED_STATUS]


def _record_filters(record: dict[str, Any]) -> dict[str, str]:
    """Translate benchmark metadata into the existing retrieval filter shape."""

    fields = ("company", "ticker", "fiscal_year", "quarter", "document_type")
    return {
        field: str(record[field])
        for field in fields
        if record.get(field) not in (None, "")
    }


def _source_references(metadata: dict[str, Any]) -> Any:
    """Return source references without dropping their original representation."""

    references = metadata.get("source_references")
    if isinstance(references, str):
        try:
            return json.loads(references)
        except json.JSONDecodeError:
            return references
    return deepcopy(references or [])


def _document_output(document: dict[str, Any]) -> dict[str, Any]:
    """Serialize a retrieved document while preserving metadata and citations."""

    metadata = deepcopy(document.get("metadata") or {})
    return {
        "text": str(document.get("text") or ""),
        "distance": document.get("distance"),
        "metadata": metadata,
        "source_url": metadata.get("source_url", ""),
        "source_references": _source_references(metadata),
    }


def _source_metadata_complete(documents: list[dict[str, Any]]) -> bool:
    """Flag whether every document has enough source data for manual verification."""

    required = ("source_url", "original_title", "location")
    return bool(documents) and all(
        all(document["metadata"].get(field) for field in required)
        and bool(document["source_references"])
        for document in documents
    )


def run_pipeline_case(
    record: dict[str, Any],
    *,
    retrieval_runner: Callable[..., list[dict[str, Any]]],
    rerank_runner: Callable[..., list[dict[str, Any]]] | None,
    collection: Any,
    model: Any,
    config: Any,
    client_provider: Callable[[], Any],
    answer_runner: Callable[..., str],
    history_provider: Callable[[str], list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Run one question through the existing RAG subgraph in one retrieval mode."""

    from rag_graph import build_rag_graph

    graph = build_rag_graph(
        collection=collection,
        model=model,
        config=config,
        retrieval_runner=retrieval_runner,
        rerank_runner=rerank_runner,
        client_provider=client_provider,
        answer_runner=answer_runner,
        history_provider=history_provider,
    )
    state = graph.invoke(
        {"question": record["question"], "filters": _record_filters(record)}
    )
    documents = [
        _document_output(document)
        for document in (state.get("document_evidence") or state.get("retrieved_chunks") or [])
    ]
    return {
        **deepcopy(record),
        "retrieval_mode": "enhanced" if rerank_runner is not None else "baseline",
        "answer": str(state.get("answer") or ""),
        "retrieved_contexts": [document["text"] for document in documents],
        "retrieved_documents": documents,
        "sources": deepcopy(state.get("sources") or []),
        "source_metadata_complete": _source_metadata_complete(documents),
        "scoring_status": (
            "eligible" if record.get("verification_status") == VERIFIED_STATUS else PENDING_STATUS
        ),
        "evaluation_error": None,
    }


def run_benchmark(
    records: list[dict[str, Any]],
    *,
    collection: Any,
    model: Any,
    config: Any,
    retrieval_runner: Callable[..., list[dict[str, Any]]],
    rerank_runner: Callable[..., list[dict[str, Any]]] | None,
    client_provider: Callable[[], Any],
    answer_runner: Callable[..., str],
    history_provider: Callable[[str], list[dict[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    """Run the same benchmark records with either baseline or enhanced retrieval."""

    results = []
    for record in records:
        try:
            results.append(
                run_pipeline_case(
                    record,
                    retrieval_runner=retrieval_runner,
                    rerank_runner=rerank_runner,
                    collection=collection,
                    model=model,
                    config=config,
                    client_provider=client_provider,
                    answer_runner=answer_runner,
                    history_provider=history_provider,
                )
            )
        except Exception as error:  # Keep one failed case from hiding the rest of the report.
            results.append(
                {
                    **deepcopy(record),
                    "retrieval_mode": "enhanced" if rerank_runner is not None else "baseline",
                    "answer": "",
                    "retrieved_contexts": [],
                    "retrieved_documents": [],
                    "sources": [],
                    "source_metadata_complete": False,
                    "scoring_status": "pipeline_error",
                    "evaluation_error": f"{type(error).__name__}: {error}",
                }
            )
    return results


def build_ragas_metrics(llm: Any, embeddings: Any) -> list[Any]:
    """Build the four Ragas 0.4.0 metrics used by this evaluation."""

    metric_classes = _load_ragas_components()[2]
    Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall = metric_classes
    return [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]


def _load_ragas_components() -> tuple[Any, Any, tuple[Any, Any, Any, Any], Any]:
    """Load Ragas lazily so the live application never imports the evaluator."""

    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    return (
        EvaluationDataset,
        SingleTurnSample,
        (Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall),
        evaluate,
    )


def score_with_ragas(
    results: list[dict[str, Any]], *, llm: Any, embeddings: Any
) -> list[dict[str, Any]]:
    """Score eligible results and leave pending/error records explicitly unscored."""

    eligible = [
        result
        for result in results
        if result.get("scoring_status") == "eligible"
        and result.get("answer")
        and result.get("retrieved_contexts")
    ]
    for result in results:
        result.setdefault("faithfulness", None)
        result.setdefault("answer_relevancy", None)
        result.setdefault("context_precision", None)
        result.setdefault("context_recall", None)
    if not eligible:
        return results

    try:
        EvaluationDataset, SingleTurnSample, _metric_classes, evaluate = _load_ragas_components()
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=result["question"],
                    response=result["answer"],
                    retrieved_contexts=result["retrieved_contexts"],
                    reference=result["ground_truth"],
                )
                for result in eligible
            ]
        )
        evaluation_result = evaluate(
            dataset,
            metrics=build_ragas_metrics(llm, embeddings),
            raise_exceptions=False,
            show_progress=False,
        )
        scored = evaluation_result.to_pandas().to_dict(orient="records")
        metric_columns = {
            "faithfulness": ("faithfulness",),
            "answer_relevancy": ("answer_relevancy",),
            "context_precision": (
                "context_precision",
                "llm_context_precision_without_reference",
            ),
            "context_recall": ("context_recall",),
        }
        for result, row in zip(eligible, scored):
            for output_name, possible_columns in metric_columns.items():
                for column in possible_columns:
                    if column in row:
                        result[output_name] = row[column]
                        break
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        logger.exception("Ragas evaluation failed")
        for result in eligible:
            result["scoring_status"] = "evaluation_error"
            result["evaluation_error"] = message
    return results


def write_evaluation_reports(results: list[dict[str, Any]], output_dir: str | Path) -> dict[str, list[Path]]:
    """Write one JSON and one CSV report for each retrieval mode."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, list[Path]] = {}
    for mode in ("baseline", "enhanced"):
        mode_results = [result for result in results if result.get("retrieval_mode") == mode]
        json_path = destination / f"{mode}_results.json"
        csv_path = destination / f"{mode}_results.csv"
        json_path.write_text(json.dumps(mode_results, indent=2, default=str), encoding="utf-8")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REPORT_FIELDS))
            writer.writeheader()
            for result in mode_results:
                writer.writerow(
                    {
                        field: json.dumps(result.get(field), default=str)
                        if isinstance(result.get(field), (dict, list))
                        else result.get(field)
                        for field in REPORT_FIELDS
                    }
                )
        paths[mode] = [json_path, csv_path]
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Equity Lens Ragas evaluation.")
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/benchmark_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results"))
    return parser.parse_args()


def main() -> None:
    """Run the explicit offline evaluator against the local indexed collection."""

    args = _parse_args()
    from functools import partial

    from rag_graph import build_rag_graph  # noqa: F401 - verifies the evaluation path import
    from stock_rag_chatbot import (
        answer_with_rag,
        get_openai_client,
        load_chroma_collection,
        load_configuration,
        load_embedding_model,
        rerank_retrieved_chunks,
        retrieve_relevant_chunks,
    )

    records = load_benchmark_dataset(args.benchmark)
    config = load_configuration()
    collection = load_chroma_collection(str(config.chroma_db_path.resolve()))
    model = load_embedding_model(config.embedding_model)
    common = {
        "collection": collection,
        "model": model,
        "config": config,
        "retrieval_runner": retrieve_relevant_chunks,
        "client_provider": lambda: get_openai_client(config),
        "answer_runner": answer_with_rag,
    }
    baseline = run_benchmark(**common, records=records, rerank_runner=None)
    enhanced = run_benchmark(
        **common,
        records=records,
        rerank_runner=partial(rerank_retrieved_chunks, config=config),
    )
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from openai import AsyncOpenAI

    async_client = AsyncOpenAI(api_key=config.openai_api_key)
    ragas_llm = llm_factory(config.openai_model, client=async_client)
    ragas_embeddings = embedding_factory("openai", model="text-embedding-3-small", client=async_client)
    all_results = score_with_ragas(baseline, llm=ragas_llm, embeddings=ragas_embeddings)
    all_results.extend(score_with_ragas(enhanced, llm=ragas_llm, embeddings=ragas_embeddings))
    paths = write_evaluation_reports(all_results, args.output)
    for mode, mode_paths in paths.items():
        print(f"{mode}: {', '.join(str(path) for path in mode_paths)}")


if __name__ == "__main__":
    main()
