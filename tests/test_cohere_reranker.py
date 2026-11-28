"""Offline tests for optional Cohere reranking and RAG integration."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import stock_rag_chatbot as chatbot
from cohere_reranker import rerank_documents
from rag_graph import build_rag_graph


def make_document(text, ticker, source_url):
    return {
        "text": text,
        "metadata": {
            "company": "Example Limited",
            "ticker": ticker,
            "exchange": "NSE",
            "fiscal_year": "FY2026",
            "quarter": "Q1",
            "document_type": "Financial Results",
            "publication_date": "2026-07-15",
            "source_url": source_url,
            "source_references": [
                {
                    "exchange": "NSE",
                    "title": "Quarterly Results",
                    "publication_date": "2026-07-15",
                    "source_url": source_url,
                }
            ],
            "location": "page 12",
        },
        "distance": 0.1,
    }


class CohereRerankerTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            make_document("Revenue grew by 10 percent.", "EXAMPLE", "https://nse.test/1"),
            make_document("The board approved a dividend.", "EXAMPLE", "https://nse.test/2"),
            make_document("Profit rose by 15 percent.", "EXAMPLE", "https://nse.test/3"),
        ]

    def make_client(self, indices):
        client = Mock()
        client.rerank.return_value = SimpleNamespace(
            results=[SimpleNamespace(index=index, relevance_score=0.9) for index in indices]
        )
        return client

    def test_successful_reranking_orders_documents_and_limits_top_k(self):
        client = self.make_client([2, 0])
        with patch("cohere_reranker._create_client", return_value=client):
            ranked = rerank_documents(
                "How did profit change?",
                self.documents,
                2,
                api_key="test-key",
                model="rerank-v5.0-fast",
            )

        self.assertEqual(ranked, [self.documents[2], self.documents[0]])
        self.assertIs(ranked[0], self.documents[2])
        client.rerank.assert_called_once_with(
            model="rerank-v5.0-fast",
            query="How did profit change?",
            documents=[document["text"] for document in self.documents],
            top_n=2,
        )

    def test_reranking_preserves_all_original_metadata_and_citations(self):
        client = self.make_client([1])
        with patch("cohere_reranker._create_client", return_value=client):
            ranked = rerank_documents(
                "What was announced?",
                self.documents,
                1,
                api_key="test-key",
                model="rerank-v5.0-fast",
            )

        self.assertIs(ranked[0], self.documents[1])
        self.assertIs(ranked[0]["metadata"], self.documents[1]["metadata"])
        self.assertEqual(ranked[0]["metadata"]["source_url"], "https://nse.test/2")
        self.assertEqual(ranked[0]["metadata"]["source_references"][0]["source_url"], "https://nse.test/2")
        self.assertEqual(ranked[0]["metadata"]["location"], "page 12")

    def test_missing_api_key_falls_back_to_chroma_order_without_client(self):
        with patch("cohere_reranker._create_client") as create_client:
            ranked = rerank_documents(
                "Question",
                self.documents,
                2,
                api_key="",
                model="rerank-v5.0-fast",
            )

        self.assertEqual(ranked, self.documents[:2])
        self.assertIs(ranked[0], self.documents[0])
        create_client.assert_not_called()

    def test_cohere_exception_falls_back_to_chroma_order(self):
        with patch(
            "cohere_reranker._create_client",
            side_effect=RuntimeError("remote API failed"),
        ):
            ranked = rerank_documents(
                "Question",
                self.documents,
                2,
                api_key="test-key",
                model="rerank-v5.0-fast",
            )

        self.assertEqual(ranked, self.documents[:2])

    def test_timeout_falls_back_to_chroma_order(self):
        client = Mock()
        client.rerank.side_effect = TimeoutError("request timed out")
        with patch("cohere_reranker._create_client", return_value=client):
            ranked = rerank_documents(
                "Question",
                self.documents,
                2,
                api_key="test-key",
                model="rerank-v5.0-fast",
            )

        self.assertEqual(ranked, self.documents[:2])

    def test_invalid_response_falls_back_to_chroma_order(self):
        client = Mock()
        client.rerank.return_value = SimpleNamespace(results=[SimpleNamespace(index=99)])
        with patch("cohere_reranker._create_client", return_value=client):
            ranked = rerank_documents(
                "Question",
                self.documents,
                2,
                api_key="test-key",
                model="rerank-v5.0-fast",
            )

        self.assertEqual(ranked, self.documents[:2])

    def test_empty_retrieval_stays_empty_without_calling_cohere(self):
        with patch("cohere_reranker._create_client") as create_client:
            ranked = rerank_documents(
                "Question",
                [],
                2,
                api_key="test-key",
                model="rerank-v5.0-fast",
            )

        self.assertEqual(ranked, [])
        create_client.assert_not_called()

    def test_load_configuration_prioritizes_new_settings_then_legacy_top_k(self):
        with (
            patch.object(chatbot, "load_dotenv"),
            patch.dict(
                os.environ,
                {
                    "TOP_K": "7",
                    "RAG_RETRIEVAL_K": "20",
                    "RAG_FINAL_K": "4",
                    "COHERE_API_KEY": "test-key",
                },
                clear=True,
            ),
        ):
            config = chatbot.load_configuration()

        self.assertEqual(config.rag_retrieval_k, 20)
        self.assertEqual(config.rag_final_k, 4)
        self.assertEqual(config.top_k, 7)
        self.assertEqual(config.cohere_api_key, "test-key")
        self.assertEqual(config.cohere_rerank_model, "rerank-v5.0-fast")

        with (
            patch.object(chatbot, "load_dotenv"),
            patch.dict(os.environ, {"TOP_K": "6"}, clear=True),
        ):
            legacy_config = chatbot.load_configuration()

        self.assertEqual(legacy_config.rag_retrieval_k, 6)
        self.assertEqual(legacy_config.rag_final_k, 6)

    def test_rag_subgraph_reranks_once_before_answer_and_keeps_source(self):
        candidates = self.documents
        selected = [self.documents[2], self.documents[0]]
        collection = Mock()
        model = object()
        retrieval_runner = Mock(return_value=candidates)
        rerank_runner = Mock(return_value=selected)
        answer_runner = Mock(return_value="Profit rose.")
        graph = build_rag_graph(
            collection=collection,
            model=model,
            config=SimpleNamespace(top_k=3, rag_retrieval_k=20),
            retrieval_runner=retrieval_runner,
            rerank_runner=rerank_runner,
            client_provider=Mock(return_value=object()),
            answer_runner=answer_runner,
        )

        state = graph.invoke({"question": "How did profit change?"})

        retrieval_runner.assert_called_once_with(
            "How did profit change?",
            collection,
            model,
            20,
            filters={},
        )
        rerank_runner.assert_called_once_with("How did profit change?", candidates)
        self.assertEqual(answer_runner.call_args.args[1], selected)
        self.assertEqual(state["sources"][0]["source_url"], "https://nse.test/3")
        self.assertEqual(state["answer"], "Profit rose.")

    def test_process_user_question_reranks_once_after_retrieval(self):
        config = SimpleNamespace(
            rag_retrieval_k=20,
            rag_final_k=1,
            cohere_api_key="",
            cohere_rerank_model="rerank-v5.0-fast",
        )
        collection = Mock()
        collection.count.return_value = 1
        with (
            patch.object(chatbot, "retrieve_relevant_chunks", return_value=self.documents) as retrieve,
            patch.object(chatbot, "rerank_documents", return_value=[self.documents[2]]) as rerank,
            patch.object(chatbot, "get_openai_client", return_value=Mock()),
            patch.object(chatbot, "answer_with_rag", return_value="Profit answer."),
        ):
            result = chatbot.process_user_question(
                "What was the profit change?", collection, object(), [], config
            )

        retrieve.assert_called_once_with(
            "What was the profit change?", collection, unittest.mock.ANY, 20, filters=None
        )
        rerank.assert_called_once_with(
            "What was the profit change?",
            self.documents,
            1,
            api_key="",
            model="rerank-v5.0-fast",
        )
        self.assertEqual(result["sources"][0]["source_url"], "https://nse.test/3")


if __name__ == "__main__":
    unittest.main()
