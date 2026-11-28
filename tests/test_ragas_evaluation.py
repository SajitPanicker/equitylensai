"""Offline tests for the isolated Ragas evaluation workflow."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluation.ragas_evaluation import (
    BenchmarkValidationError,
    build_ragas_metrics,
    evaluation_records,
    load_benchmark_dataset,
    run_benchmark,
    score_with_ragas,
    validate_benchmark_record,
    write_evaluation_reports,
)


def benchmark_record(**overrides):
    record = {
        "question": "What was revenue growth?",
        "ground_truth": "Revenue grew in the verified filing.",
        "company": "Example Limited",
        "ticker": "EXAMPLE",
        "fiscal_year": "FY2024",
        "quarter": None,
        "document_type": "annual_report",
        "reference_sources": [{"source_url": "https://nse.test/report.pdf", "page": 12}],
        "verification_status": "verified",
    }
    record.update(overrides)
    return record


def document(text="Revenue grew."):
    return {
        "text": text,
        "distance": 0.2,
        "metadata": {
            "company": "Example Limited",
            "ticker": "EXAMPLE",
            "fiscal_year": "FY2024",
            "original_title": "Annual Report",
            "location": "page 12",
            "source_url": "https://nse.test/report.pdf",
            "source_references": [
                {
                    "exchange": "NSE",
                    "title": "Annual Report",
                    "publication_date": "2024-05-01",
                    "source_url": "https://nse.test/report.pdf",
                }
            ],
        },
    }


class RagasEvaluationTests(unittest.TestCase):
    def test_dataset_loading_and_pending_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_text(
                json.dumps([benchmark_record(ground_truth=None, reference_sources=[], verification_status="pending_verification")]),
                encoding="utf-8",
            )

            records = load_benchmark_dataset(path)

        self.assertEqual(records[0]["verification_status"], "pending_verification")

    def test_verified_record_missing_ground_truth_is_rejected(self):
        with self.assertRaises(BenchmarkValidationError):
            validate_benchmark_record(benchmark_record(ground_truth=None))

    def test_pipeline_supports_baseline_and_reranked_modes(self):
        candidates = [document("First."), document("Second.")]
        retrieval = Mock(return_value=candidates)
        answer = Mock(return_value="Verified answer.")
        common = {
            "collection": Mock(),
            "model": object(),
            "config": SimpleNamespace(top_k=3, rag_retrieval_k=3, rag_final_k=2),
            "retrieval_runner": retrieval,
            "client_provider": Mock(return_value=object()),
            "answer_runner": answer,
        }

        baseline = run_benchmark([benchmark_record()], **common, rerank_runner=None)
        reranker = Mock(return_value=[candidates[1], candidates[0]])
        enhanced = run_benchmark([benchmark_record()], **common, rerank_runner=reranker)

        self.assertEqual(baseline[0]["retrieval_mode"], "baseline")
        self.assertEqual(enhanced[0]["retrieval_mode"], "enhanced")
        self.assertEqual(enhanced[0]["retrieved_contexts"], ["Second.", "First."])
        self.assertEqual(
            enhanced[0]["retrieved_documents"][0]["source_references"][0]["source_url"],
            "https://nse.test/report.pdf",
        )
        reranker.assert_called_once()

    def test_pending_records_are_excluded_before_retrieval_by_default(self):
        pending = benchmark_record(
            ground_truth=None, reference_sources=[], verification_status="pending_verification"
        )
        retrieval = Mock()
        answer = Mock()

        results = run_benchmark(
            [pending],
            collection=Mock(),
            model=object(),
            config=SimpleNamespace(top_k=3, rag_retrieval_k=3, rag_final_k=2),
            retrieval_runner=retrieval,
            rerank_runner=None,
            client_provider=Mock(),
            answer_runner=answer,
        )

        self.assertEqual(results, [])
        retrieval.assert_not_called()
        answer.assert_not_called()

    def test_include_pending_runs_retrieval_only_without_ragas_scoring(self):
        pending = benchmark_record(
            ground_truth=None, reference_sources=[], verification_status="pending_verification"
        )
        retrieval = Mock(return_value=[document()])
        answer = Mock(return_value="Unverified answer.")

        results = run_benchmark(
            [pending],
            collection=Mock(),
            model=object(),
            config=SimpleNamespace(top_k=3, rag_retrieval_k=3, rag_final_k=2),
            retrieval_runner=retrieval,
            rerank_runner=None,
            client_provider=Mock(return_value=object()),
            answer_runner=answer,
            include_pending=True,
        )

        self.assertEqual(results[0]["scoring_status"], "retrieval_only")
        with patch("evaluation.ragas_evaluation._load_ragas_components") as load_components:
            scored = score_with_ragas(results, llm=object(), embeddings=object())
        load_components.assert_not_called()
        self.assertIsNone(scored[0]["context_precision"])

    def test_evaluation_record_selection_keeps_verified_and_optionally_pending(self):
        verified = benchmark_record()
        pending = benchmark_record(
            ground_truth=None, reference_sources=[], verification_status="pending_verification"
        )
        self.assertEqual(evaluation_records([verified, pending]), [verified])
        self.assertEqual(evaluation_records([verified, pending], include_pending=True), [verified, pending])

    def test_missing_retrieved_context_is_not_sent_to_ragas(self):
        result = {
            **benchmark_record(),
            "retrieval_mode": "baseline",
            "answer": "An answer.",
            "retrieved_contexts": [],
        }
        with patch("evaluation.ragas_evaluation._load_ragas_components") as load_components:
            scored = score_with_ragas([result], llm=object(), embeddings=object())

        load_components.assert_not_called()
        self.assertIsNone(scored[0]["faithfulness"])
        self.assertIsNone(scored[0]["context_recall"])

    def test_ragas_metric_configuration_uses_four_metrics(self):
        metric_classes = tuple(Mock() for _ in range(4))
        with patch(
            "evaluation.ragas_evaluation._load_ragas_components",
            return_value=(object(), object(), metric_classes, Mock()),
        ):
            metrics = build_ragas_metrics("llm", "embeddings")

        self.assertEqual(len(metrics), 4)
        metric_classes[0].assert_called_once_with(llm="llm")
        metric_classes[1].assert_called_once_with(llm="llm", embeddings="embeddings")
        metric_classes[2].assert_called_once_with(llm="llm")
        metric_classes[3].assert_called_once_with(llm="llm")

    def test_context_precision_uses_reference_metric(self):
        metric_classes = tuple(Mock() for _ in range(4))
        with patch(
            "evaluation.ragas_evaluation._load_ragas_components",
            return_value=(object(), object(), metric_classes, Mock()),
        ):
            build_ragas_metrics("llm", "embeddings")

        metric_classes[2].assert_called_once_with(llm="llm")

    def test_ragas_api_failure_is_recorded_without_aborting(self):
        result = {
            **benchmark_record(),
            "retrieval_mode": "baseline",
            "answer": "An answer.",
            "retrieved_contexts": ["Evidence."],
        }
        with patch(
            "evaluation.ragas_evaluation._load_ragas_components",
            side_effect=RuntimeError("evaluator unavailable"),
        ):
            result["scoring_status"] = "eligible"
            scored = score_with_ragas([result], llm=object(), embeddings=object())

        self.assertEqual(scored[0]["scoring_status"], "evaluation_error")
        self.assertIn("evaluator unavailable", scored[0]["evaluation_error"])

    def test_ragas_scores_are_attached_to_eligible_results(self):
        result = {
            **benchmark_record(),
            "retrieval_mode": "baseline",
            "answer": "An answer.",
            "retrieved_contexts": ["Evidence."],
            "scoring_status": "eligible",
        }

        class FakeDataset:
            def __init__(self, samples):
                self.samples = samples

        class FakeSample:
            def __init__(self, **values):
                self.values = values

        class FakeEvaluationResult:
            def to_pandas(self):
                return self

            def to_dict(self, orient):
                self.assert_orient = orient
                return [
                    {
                        "faithfulness": 0.9,
                        "answer_relevancy": 0.8,
                        "llm_context_precision_with_reference": 0.7,
                        "context_recall": 0.6,
                    }
                ]

        evaluate = Mock(return_value=FakeEvaluationResult())
        metric_classes = tuple(Mock() for _ in range(4))
        with patch(
            "evaluation.ragas_evaluation._load_ragas_components",
            return_value=(FakeDataset, FakeSample, metric_classes, evaluate),
        ):
            scored = score_with_ragas([result], llm="llm", embeddings="embeddings")

        self.assertEqual(scored[0]["faithfulness"], 0.9)
        self.assertEqual(scored[0]["context_precision"], 0.7)
        self.assertEqual(scored[0]["context_recall"], 0.6)
        evaluate.assert_called_once()

        sample = evaluate.call_args.args[0].samples[0]
        self.assertEqual(sample.values["reference"], "Revenue grew in the verified filing.")

    def test_evaluation_results_are_serialized_to_json_and_csv(self):
        result = {
            **benchmark_record(),
            "retrieval_mode": "baseline",
            "answer": "An answer.",
            "retrieved_contexts": ["Evidence."],
            "retrieved_documents": [
                {
                    "text": "Evidence.",
                    "metadata": document()["metadata"],
                    "source_references": document()["metadata"]["source_references"],
                }
            ],
            "sources": [{"source_url": "https://nse.test/report.pdf"}],
            "source_metadata_complete": True,
            "scoring_status": "eligible",
            "evaluation_error": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = write_evaluation_reports([result], directory)

            json_output = json.loads((Path(directory) / "baseline_results.json").read_text())
            csv_output = (Path(directory) / "baseline_results.csv").read_text()

        self.assertEqual(len(paths["enhanced"]), 2)
        self.assertEqual(json_output[0]["retrieved_documents"][0]["metadata"]["location"], "page 12")
        self.assertEqual(json_output[0]["reference_sources"][0]["page"], 12)
        self.assertIn("reference_sources", csv_output)
        self.assertIn("https://nse.test/report.pdf", csv_output)


if __name__ == "__main__":
    unittest.main()
