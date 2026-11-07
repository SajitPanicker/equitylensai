"""Offline tests for graph routing, conversation isolation, and SQLite persistence."""

import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from equitylens_graph import build_equitylens_graph, new_thread_id


class FakeCollection:
    def __init__(self, count: int = 1) -> None:
        self.document_count = count

    def count(self) -> int:
        return self.document_count


class EquityLensGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.temporary_directory.name) / "checkpoints.sqlite")
        self.collection = FakeCollection()
        self.processor_calls: list[dict[str, Any]] = []
        self.unmatched_filter_calls: list[dict[str, str | list[str]]] = []
        self.generated_images: list[Path] = []
        self.context_manager, self.checkpointer = self.open_checkpointer()
        self.graph = self.build_graph()

    def tearDown(self) -> None:
        self.context_manager.__exit__(None, None, None)
        for image_path in self.generated_images:
            image_path.unlink(missing_ok=True)
        self.temporary_directory.cleanup()

    def open_checkpointer(self):
        context_manager = SqliteSaver.from_conn_string(self.database_path)
        return context_manager, context_manager.__enter__()

    def process_question(self, **kwargs: Any) -> dict[str, Any]:
        self.processor_calls.append(kwargs)
        if any(
            word in kwargs["question"].casefold()
            for word in ("image", "infographic", "chart")
        ):
            return {
                "answer": "Visual generated.",
                "sources": [{"label": "annual-report.pdf", "snippet": "Evidence"}],
                "image_bytes": b"test-png-bytes",
                "is_error": False,
            }
        return {
            "answer": f"Answer to: {kwargs['question']}",
            "sources": [{"label": "annual-report.pdf", "snippet": "Evidence"}],
            "image_bytes": None,
            "is_error": False,
        }

    def find_unmatched_filters(
        self, _collection: FakeCollection, filters: dict[str, str | list[str]]
    ) -> dict[str, list[str]]:
        self.assertIs(_collection, self.collection)
        self.unmatched_filter_calls.append(filters)
        return {}

    def build_graph(self):
        return build_equitylens_graph(
            collection=self.collection,
            model_provider=lambda: "mock-model",
            config="mock-config",
            process_user_question=self.process_question,
            find_unmatched_filter_values=self.find_unmatched_filters,
            checkpointer=self.checkpointer,
        )

    def invoke(
        self,
        thread_id: str,
        question: str,
        filters: dict[str, str | list[str]] | None = None,
    ) -> dict[str, Any]:
        return self.graph.invoke(
            {
                "question": question,
                "messages": [HumanMessage(content=question)],
                "filters": filters or {},
            },
            config={"configurable": {"thread_id": thread_id}},
        )

    def test_follow_up_reuses_history_and_passes_company_year_filters(self) -> None:
        thread_id = new_thread_id()
        filters = {"company": "Tata Consultancy Services", "fiscal_year": "FY2026"}

        self.invoke(thread_id, "What was revenue?", filters)
        second_result = self.invoke(thread_id, "How did it compare?", filters)

        self.assertEqual(len(self.processor_calls), 2)
        self.assertEqual(self.processor_calls[0]["filters"], filters)
        self.assertEqual(
            self.processor_calls[1]["history"],
            [
                {"role": "user", "content": "What was revenue?"},
                {"role": "assistant", "content": "Answer to: What was revenue?"},
            ],
        )
        self.assertEqual(second_result["answer"], "Answer to: How did it compare?")
        self.assertEqual(self.unmatched_filter_calls, [filters, filters])

    def test_no_indexed_documents_skips_question_processor(self) -> None:
        self.collection.document_count = 0

        result = self.invoke(new_thread_id(), "Summarize the filings")

        self.assertIn("No exchange filings are indexed yet", result["answer"])
        self.assertEqual(self.processor_calls, [])

    def test_empty_question_is_validated_without_processing(self) -> None:
        result = self.invoke(new_thread_id(), "   ")

        self.assertTrue(result["is_error"])
        self.assertIn("Enter a question", result["answer"])
        self.assertEqual(self.processor_calls, [])

    def test_no_relevant_chunks_result_is_returned_unchanged(self) -> None:
        def no_relevant_chunks(**kwargs: Any) -> dict[str, Any]:
            self.processor_calls.append(kwargs)
            return {
                "answer": "No indexed documents match the requested filters.",
                "sources": [],
                "image_bytes": None,
                "is_error": False,
            }

        self.process_question = no_relevant_chunks
        self.graph = self.build_graph()

        result = self.invoke(new_thread_id(), "What happened?", {"company": "Unknown"})

        self.assertEqual(
            result["answer"], "No indexed documents match the requested filters."
        )
        self.assertEqual(result["sources"], [])

    def test_new_thread_does_not_share_messages_or_results(self) -> None:
        first_thread = new_thread_id()
        second_thread = new_thread_id()
        self.invoke(first_thread, "First conversation question")

        result = self.invoke(second_thread, "Separate conversation question")

        messages = result["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "Separate conversation question")
        self.assertEqual(
            messages[1].content, "Answer to: Separate conversation question"
        )

    def test_image_bytes_are_stored_as_a_file_reference_and_resume_after_restart(self) -> None:
        thread_id = new_thread_id()
        result = self.invoke(thread_id, "Create an infographic image")
        image_path = result["image_path"]

        self.assertIsNotNone(image_path)
        self.assertEqual(Path(image_path).read_bytes(), b"test-png-bytes")
        self.generated_images.append(Path(image_path))
        self.assertNotIn("image_bytes", result["result"])
        self.assertEqual(result["messages"][-1].additional_kwargs["image_path"], image_path)

        self.context_manager.__exit__(None, None, None)
        self.context_manager, self.checkpointer = self.open_checkpointer()
        self.graph = self.build_graph()
        resumed_state = self.graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        ).values

        self.assertEqual(resumed_state["answer"], "Visual generated.")
        self.assertEqual(resumed_state["image_path"], image_path)
        self.assertEqual(len(resumed_state["messages"]), 2)

    def test_chart_and_infographic_results_use_the_same_processing_node(self) -> None:
        thread_id = new_thread_id()
        chart_result = self.invoke(thread_id, "Create a chart")
        infographic_result = self.invoke(thread_id, "Create an infographic")

        self.assertEqual(
            [call["question"] for call in self.processor_calls],
            ["Create a chart", "Create an infographic"],
        )
        self.assertTrue(all(call["model"] == "mock-model" for call in self.processor_calls))
        for result in (chart_result, infographic_result):
            image_path = Path(result["image_path"])
            self.assertEqual(image_path.read_bytes(), b"test-png-bytes")
            self.generated_images.append(image_path)


if __name__ == "__main__":
    unittest.main()
