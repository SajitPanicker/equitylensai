"""Offline tests for parent-graph intent routing and checkpointed conversations."""

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langgraph.checkpoint.sqlite import SqliteSaver

import stock_rag_chatbot as chatbot
from equitylens_graph import build_equitylens_graph, route_by_intent
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
        graph = build_rag_graph(
            collection=collection,
            model=model,
            config=config,
            history_provider=Mock(return_value=history),
            answer_runner=Mock(return_value="Revenue increased."),
        )

        with (
            patch("rag_graph.retrieve_relevant_chunks", return_value=chunks) as retrieve,
            patch("rag_graph.get_openai_client", return_value=object()),
        ):
            state = graph.invoke(
                {"question": "Summarize Infosys revenue", "filters": {"ticker": "INFY.NS"}}
            )

        retrieve.assert_called_once_with(
            "Summarize Infosys revenue",
            collection,
            model,
            3,
            filters={"ticker": "INFY.NS"},
        )
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
        graph = build_rag_graph(
            collection=collection,
            model=model,
            config=config,
            defer_answer=True,
            answer_runner=answer_runner,
        )

        with patch("rag_graph.retrieve_relevant_chunks", return_value=chunks):
            state = graph.invoke({"question": "Summarize FY2025 net income"})

        self.assertEqual(state["document_evidence"], chunks)
        self.assertEqual(state["answer"], "")
        answer_runner.assert_not_called()

    def test_rag_retrieval_error_is_explicit_and_returns_no_evidence(self):
        graph = build_rag_graph(
            collection=Mock(),
            model=object(),
            config=SimpleNamespace(top_k=3),
            defer_answer=True,
        )
        with patch(
            "rag_graph.retrieve_relevant_chunks",
            side_effect=RuntimeError("ChromaDB unavailable"),
        ):
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


if __name__ == "__main__":
    unittest.main()
