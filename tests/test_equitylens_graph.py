"""Offline tests for parent-graph intent routing and checkpointed conversations."""

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

import stock_rag_chatbot as chatbot
from equitylens_graph import build_equitylens_graph, new_thread_id, route_by_intent
from rag_graph import build_rag_graph


class StaticSubgraph:
    def __init__(self, answer, delay=0, is_error=False):
        self.answer = answer
        self.delay = delay
        self.is_error = is_error

    def invoke(self, state):
        if self.delay:
            time.sleep(self.delay)
        result = {
            "answer": self.answer,
            "sources": [{"label": self.answer, "snippet": state.get("question", "")}],
            "is_error": self.is_error,
        }
        if self.is_error:
            result["error_message"] = self.answer
            return result
        if self.answer == "filing evidence":
            result["document_evidence"] = [
                {
                    "text": "Revenue increased in FY2025.",
                    "metadata": {
                        "company": "Infosys",
                        "ticker": "INFY.NS",
                        "exchange": "NSE",
                        "fiscal_year": "FY2025",
                        "original_title": "Annual Report",
                        "location": "page 12",
                    },
                }
            ]
        if self.answer == "Yahoo Finance data":
            result["market_data"] = {
                "company_name": "Infosys",
                "ticker": "INFY.NS",
                "latest_market_data": {"latest_available_price": 1500.0},
                "historical_returns": {"annual_returns": [{"year": 2025, "return_pct": 12.5}]},
                "source": "Yahoo Finance via yfinance",
                "data_timestamp": "2026-11-08T10:00:00+00:00",
            }
        return result


class EquityLensGraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "checkpoints.sqlite"
        self.collection = Mock()
        self.collection.count.return_value = 1
        self.config = SimpleNamespace(openai_model="gpt-4o-mini")

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def classification(analysis_type):
        return {
            "company_name": "Infosys",
            "ticker": "INFY.NS",
            "analysis_type": analysis_type,
            "use_rag": analysis_type in {"rag", "both"},
            "use_yahoo_finance": analysis_type in {"yahoo", "both"},
            "classification_reason": "Test classification",
        }

    def build_graph(
        self,
        saver,
        classification,
        rag_delay=0,
        yahoo_delay=0,
        rag_error=False,
        yahoo_error=False,
    ):
        def rag_builder(**kwargs):
            return StaticSubgraph("RAG retrieval failed" if rag_error else "filing evidence", rag_delay, rag_error)

        def yahoo_builder():
            return StaticSubgraph("Yahoo Finance unavailable" if yahoo_error else "Yahoo Finance data", yahoo_delay, yahoo_error)

        self.financial_analysis_runner = Mock(return_value="LLM financial synthesis")
        return build_equitylens_graph(
            collection=self.collection,
            model_provider=lambda: Mock(),
            embedding_model_provider=lambda: object(),
            config=self.config,
            process_user_question=lambda **kwargs: {
                "answer": "filing evidence",
                "sources": [],
                "is_error": False,
            },
            find_unmatched_filter_values=lambda collection, filters: {},
            checkpointer=saver,
            classification_helper=lambda question, config, client: classification,
            financial_analysis_runner=self.financial_analysis_runner,
            is_image_generation_request=chatbot.is_image_generation_request,
            needs_visual_clarification=chatbot.needs_visual_clarification,
            rag_graph_builder=rag_builder,
            yahoo_graph_builder=yahoo_builder,
        )

    def invoke(self, graph, question, thread_id="thread-a"):
        return graph.invoke(
            {"question": question},
            config={"configurable": {"thread_id": thread_id}},
        )

    def test_classifier_helper_returns_validated_structured_output(self):
        fake_client = Mock()
        fake_client.chat.completions.create.return_value.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"company_name":"Infosys","ticker":"INFY.NS",'
                        '"analysis_type":"both","classification_reason":"Needs filings and market data"}'
                    )
                )
            )
        ]
        with patch.object(chatbot, "get_openai_client", return_value=fake_client):
            result = chatbot.classify_financial_intent(
                "Compare Infosys filings with current stock valuation",
                SimpleNamespace(openai_model="gpt-4o-mini"),
            )

        self.assertEqual(result["analysis_type"], "both")
        self.assertTrue(result["use_rag"])
        self.assertTrue(result["use_yahoo_finance"])
        self.assertEqual(result["company_name"], "Infosys")
        fake_client.chat.completions.create.assert_called_once()

    def test_classifier_helper_rejects_invalid_intent(self):
        fake_client = Mock()
        fake_client.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content='{"analysis_type":"unknown"}'))
        ]
        with (
            patch.object(chatbot, "get_openai_client", return_value=fake_client),
            self.assertRaises(ValueError),
        ):
            chatbot.classify_financial_intent(
                "What is happening?",
                SimpleNamespace(openai_model="gpt-4o-mini"),
            )

    def test_financial_context_llm_prompt_contains_selected_source_data(self):
        fake_client = Mock()
        fake_client.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content="Financial assessment"))
        ]
        evidence = [
            {
                "text": "Revenue increased in FY2025.",
                "metadata": {
                    "company": "Infosys",
                    "exchange": "NSE",
                    "fiscal_year": "FY2025",
                    "original_title": "Annual Report",
                    "location": "page 12",
                },
            }
        ]
        market_data = {
            "ticker": "INFY.NS",
            "source": "Yahoo Finance via yfinance",
            "data_timestamp": "2026-11-08T10:00:00+00:00",
            "latest_market_data": {"latest_available_price": 1500.0},
            "historical_returns": {"annual_returns": [{"year": 2025, "return_pct": 12.5}]},
        }

        answer = chatbot.answer_with_financial_context(
            "Compare revenue with share performance",
            [{"role": "user", "content": "Earlier question"}],
            evidence,
            market_data,
            [],
            SimpleNamespace(openai_model="gpt-4o-mini"),
            fake_client,
        )

        prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
        self.assertEqual(answer, "Financial assessment")
        self.assertIn("Compare revenue with share performance", prompt)
        self.assertIn("Revenue increased in FY2025.", prompt)
        self.assertIn("official_nse_bse_document_evidence", prompt)
        self.assertIn("Yahoo Finance via yfinance", prompt)
        self.assertIn("1500.0", prompt)
        self.assertIn("historical_returns", prompt)
        self.assertIn("Earlier question", str(fake_client.chat.completions.create.call_args.kwargs["messages"]))

    def test_yahoo_price_summary_includes_year_by_year_closes_and_signed_changes(self):
        answer = chatbot.summarize_yahoo_market_question(
            "I want year by year stock price of Infosys for past 5 years",
            {
                "company_name": "Infosys",
                "ticker": "INFY.NS",
                "historical_prices": {
                    "points": [
                        {"date": "2023-12-29", "close": 100.0},
                        {"date": "2024-12-31", "close": 80.0},
                        {"date": "2025-11-28", "close": 90.0},
                    ]
                },
            },
        )

        self.assertIn("total change of -10.00%", answer)
        self.assertIn("2023 year-end close: ₹100.00", answer)
        self.assertIn("2024 year-end close: ₹80.00 (-20.00% vs 2023)", answer)
        self.assertIn("2025 latest monthly close through 2025-11-28: ₹90.00 (+12.50% vs 2024)", answer)
        self.assertIn("latest year is year-to-date", answer)

    def test_yahoo_price_summary_returns_markdown_table_when_requested(self):
        answer = chatbot.summarize_yahoo_market_question(
            "Create a table of last 5 years stock price of Infosys",
            {
                "company_name": "Infosys",
                "ticker": "INFY.NS",
                "historical_prices": {
                    "points": [
                        {"date": "2023-12-29", "close": 100.0},
                        {"date": "2024-12-31", "close": 80.0},
                        {"date": "2025-11-28", "close": 90.0},
                    ]
                },
            },
        )

        self.assertIn("| Year | Price date | Closing price | Change vs prior year |", answer)
        self.assertIn("| 2023 | 2023-12-29 | ₹100.00 | N/A |", answer)
        self.assertIn("| 2024 | 2024-12-31 | ₹80.00 | -20.00% |", answer)
        self.assertIn("| 2025 | 2025-11-28 | ₹90.00 | +12.50% |", answer)

    def test_route_by_intent_for_each_valid_selection_and_invalid_state(self):
        self.assertEqual(route_by_intent({"use_rag": True, "use_yahoo_finance": False}), "rag")
        self.assertEqual(route_by_intent({"use_rag": False, "use_yahoo_finance": True}), "yahoo")
        self.assertEqual(route_by_intent({"use_rag": True, "use_yahoo_finance": True}), "both")
        self.assertEqual(route_by_intent({"use_rag": False, "use_yahoo_finance": False}), "rag")
        self.assertEqual(route_by_intent({"use_rag": "yes", "use_yahoo_finance": False}), "rag")

    def test_parent_graph_updates_intent_state_and_routes_rag_only(self):
        with SqliteSaver.from_conn_string(str(self.database_path)) as saver:
            graph = self.build_graph(saver, self.classification("rag"))
            state = self.invoke(graph, "Summarize Infosys annual filing")

        self.assertEqual(state["analysis_type"], "rag")
        self.assertEqual(state["company_name"], "Infosys")
        self.assertEqual(state["ticker"], "INFY.NS")
        self.assertTrue(state["use_rag"])
        self.assertFalse(state["use_yahoo_finance"])
        self.assertEqual(state["classification_reason"], "Test classification")
        self.assertEqual(state["answer"], "LLM financial synthesis")
        self.assertEqual(
            self.financial_analysis_runner.call_args.args[2][0]["text"],
            "Revenue increased in FY2025.",
        )
        self.assertEqual(self.financial_analysis_runner.call_args.args[3], {})

    def test_parent_graph_routes_yahoo_only(self):
        with SqliteSaver.from_conn_string(str(self.database_path)) as saver:
            graph = self.build_graph(saver, self.classification("yahoo"))
            state = self.invoke(graph, "What is Infosys share price today")

        self.assertFalse(state["use_rag"])
        self.assertTrue(state["use_yahoo_finance"])
        self.assertEqual(state["answer"], "LLM financial synthesis")
        self.assertEqual(self.financial_analysis_runner.call_args.args[2], [])
        self.assertEqual(
            self.financial_analysis_runner.call_args.args[3]["ticker"],
            "INFY.NS",
        )

    def test_both_parallel_branches_finish_before_synthesis(self):
        with SqliteSaver.from_conn_string(str(self.database_path)) as saver:
            graph = self.build_graph(
                saver,
                self.classification("both"),
                rag_delay=0.05,
                yahoo_delay=0.15,
            )
            state = self.invoke(graph, "Compare Infosys filings with current market data")

        self.assertEqual(state["answer"], "LLM financial synthesis")
        self.assertEqual(len(state["sources"]), 2)
        self.assertTrue(self.financial_analysis_runner.called)
        self.assertEqual(len(self.financial_analysis_runner.call_args.args[2]), 1)
        self.assertEqual(
            self.financial_analysis_runner.call_args.args[3]["ticker"], "INFY.NS"
        )
        self.assertEqual(
            self.financial_analysis_runner.call_args.args[4], []
        )

    def test_missing_yahoo_data_is_disclosed_while_rag_evidence_is_analyzed(self):
        with SqliteSaver.from_conn_string(str(self.database_path)) as saver:
            graph = self.build_graph(
                saver,
                self.classification("both"),
                yahoo_error=True,
            )
            state = self.invoke(graph, "Compare Infosys filing with market performance")

        self.assertEqual(len(self.financial_analysis_runner.call_args.args[2]), 1)
        self.assertEqual(self.financial_analysis_runner.call_args.args[3], {})
        self.assertIn("Yahoo Finance market data could not be retrieved", self.financial_analysis_runner.call_args.args[4][0])
        self.assertFalse(state["result"]["is_error"])

    def test_missing_rag_evidence_is_disclosed_while_yahoo_data_is_analyzed(self):
        with SqliteSaver.from_conn_string(str(self.database_path)) as saver:
            graph = self.build_graph(
                saver,
                self.classification("both"),
                rag_error=True,
            )
            state = self.invoke(graph, "Compare Infosys filing with market performance")

        self.assertEqual(self.financial_analysis_runner.call_args.args[2], [])
        self.assertEqual(
            self.financial_analysis_runner.call_args.args[3]["ticker"], "INFY.NS"
        )
        self.assertIn("Official NSE/BSE document retrieval failed", self.financial_analysis_runner.call_args.args[4][0])
        self.assertFalse(state["result"]["is_error"])

    def test_rag_subgraph_retrieves_evidence_and_generates_an_answer(self):
        collection = Mock()
        model = object()
        config = SimpleNamespace(top_k=3)
        chunks = [
            {
                "text": "Revenue increased during the year.",
                "metadata": {
                    "company": "Infosys",
                    "ticker": "INFY.NS",
                    "fiscal_year": "FY2025",
                    "original_title": "Annual Report",
                    "location": "page 12",
                },
            }
        ]
        history = [{"role": "user", "content": "Prior question"}]
        retrieval_runner = Mock(return_value=chunks)
        client_provider = Mock(return_value=object())
        answer_runner = Mock(return_value="Revenue increased.")
        graph = build_rag_graph(
            collection=collection,
            model=model,
            config=config,
            retrieval_runner=retrieval_runner,
            client_provider=client_provider,
            history_provider=Mock(return_value=history),
            answer_runner=answer_runner,
        )

        state = graph.invoke(
            {"question": "Summarize Infosys revenue", "filters": {"ticker": "INFY.NS"}}
        )

        retrieval_runner.assert_called_once_with(
            "Summarize Infosys revenue",
            collection,
            model,
            3,
            filters={"ticker": "INFY.NS"},
        )
        client_provider.assert_called_once_with()
        answer_runner.assert_called_once()
        self.assertEqual(state["answer"], "Revenue increased.")
        self.assertEqual(len(state["sources"]), 1)
        self.assertIn("Infosys (INFY.NS)", state["sources"][0]["label"])

    def test_rag_subgraph_can_return_evidence_without_answering_early(self):
        collection = Mock()
        model = object()
        config = SimpleNamespace(top_k=3)
        chunks = [
            {
                "text": "Net income was ₹10 crore.",
                "metadata": {"exchange": "BSE", "fiscal_year": "FY2025"},
            }
        ]
        answer_runner = Mock(side_effect=AssertionError("Answer should be synthesized by the parent"))
        retrieval_runner = Mock(return_value=chunks)
        graph = build_rag_graph(
            collection=collection,
            model=model,
            config=config,
            retrieval_runner=retrieval_runner,
            client_provider=Mock(return_value=object()),
            defer_answer=True,
            answer_runner=answer_runner,
        )

        state = graph.invoke({"question": "Summarize FY2025 net income"})

        self.assertEqual(state["document_evidence"], chunks)
        self.assertEqual(state["answer"], "")
        answer_runner.assert_not_called()
        retrieval_runner.assert_called_once()

    def test_rag_retrieval_error_is_explicit_and_returns_no_evidence(self):
        retrieval_runner = Mock(side_effect=RuntimeError("ChromaDB unavailable"))
        graph = build_rag_graph(
            collection=Mock(),
            model=object(),
            config=SimpleNamespace(top_k=3),
            retrieval_runner=retrieval_runner,
            client_provider=Mock(return_value=object()),
            answer_runner=Mock(return_value=""),
            defer_answer=True,
        )
        state = graph.invoke({"question": "Summarize FY2025 net income"})

        self.assertTrue(state["is_error"])
        self.assertEqual(state["document_evidence"], [])
        self.assertIn("ChromaDB unavailable", state["error_message"])

    def test_invalid_classifier_output_falls_back_to_rag(self):
        invalid_classification = {
            "analysis_type": "unknown",
            "use_rag": False,
            "use_yahoo_finance": False,
        }
        with SqliteSaver.from_conn_string(str(self.database_path)) as saver:
            graph = self.build_graph(saver, invalid_classification)
            state = self.invoke(graph, "Explain the latest Infosys filing")

        self.assertTrue(state["use_rag"])
        self.assertFalse(state["use_yahoo_finance"])
        self.assertIsNotNone(state["classification_error"])
        self.assertIn("safe fallback", state["answer"])
        self.assertIn("LLM financial synthesis", state["answer"])

    def test_sqlite_checkpointing_and_thread_isolation(self):
        with SqliteSaver.from_conn_string(str(self.database_path)) as saver:
            graph = self.build_graph(saver, self.classification("rag"))
            first = self.invoke(graph, "Question in first conversation", "thread-one")
            second = self.invoke(graph, "Question in second conversation", "thread-two")

            first_state = graph.get_state(
                {"configurable": {"thread_id": "thread-one"}}
            ).values
            second_state = graph.get_state(
                {"configurable": {"thread_id": "thread-two"}}
            ).values

        self.assertEqual(first["answer"], second["answer"])
        self.assertEqual(first_state["question"], "Question in first conversation")
        self.assertEqual(second_state["question"], "Question in second conversation")
        self.assertNotEqual(first_state["messages"], second_state["messages"])


class FakeCollection:
    def __init__(self, count: int = 1) -> None:
        self.document_count = count

    def count(self) -> int:
        return self.document_count


class EquityLensConversationGraphTests(unittest.TestCase):
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
        self.last_processed_result = None

        def question_processor(**kwargs):
            self.last_processed_result = self.process_question(**kwargs)
            return self.last_processed_result

        def rag_builder(**options):
            if self.collection.count():
                options["question_processor"] = question_processor
            return build_rag_graph(
                **options,
                retrieval_runner=Mock(return_value=[]),
                client_provider=Mock(return_value=object()),
                answer_runner=Mock(return_value=""),
            )

        classification = {
            "company_name": "Tata Consultancy Services",
            "ticker": "TCS.NS",
            "analysis_type": "rag",
            "use_rag": True,
            "use_yahoo_finance": False,
            "classification_reason": "Test filing question",
        }

        def financial_analysis_runner(
            question, history, evidence, market, status, config, client
        ):
            if not self.collection.count():
                return "No exchange filings are indexed yet. Discover filings above and index selected documents first."
            return str((self.last_processed_result or {}).get("answer") or "")

        return build_equitylens_graph(
            collection=self.collection,
            model_provider=lambda: "mock-model",
            config="mock-config",
            process_user_question=self.process_question,
            find_unmatched_filter_values=self.find_unmatched_filters,
            checkpointer=self.checkpointer,
            classification_helper=lambda question, config, client: classification,
            financial_analysis_runner=financial_analysis_runner,
            is_image_generation_request=chatbot.is_image_generation_request,
            needs_visual_clarification=chatbot.needs_visual_clarification,
            rag_graph_builder=rag_builder,
            yahoo_graph_builder=lambda: None,
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
