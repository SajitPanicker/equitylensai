"""Fast tests for text/image routing; these tests do not call OpenAI."""

import base64
import json
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from openai import APIConnectionError
from PIL import Image

import stock_rag_chatbot as chatbot


class ImageRoutingTests(unittest.TestCase):
    """Verify request routing, evidence retrieval, and safe API error responses."""

    def setUp(self) -> None:
        # These simple values let tests exercise routing without real files or secrets.
        self.config = chatbot.AppConfig(
            openai_api_key="test-key-not-a-real-secret",
            documents_folder=chatbot.Path("presentations"),
            chroma_db_path=chatbot.Path("test_chroma"),
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            openai_model="gpt-4o-mini",
            image_model="gpt-image-2.5-flare",
            image_size="1024x1536",
            image_quality="medium",
            top_k=4,
            chunk_size=1000,
            chunk_overlap=150,
        )
        # The fake collection represents an already-indexed document store.
        self.collection = Mock()
        self.collection.count.return_value = 1
        # The fake match looks like a result returned by the existing ChromaDB search.
        self.matches = [
            {
                "text": "Revenue was INR 100 crore in FY2025.",
                "metadata": {
                    "source": "annual-report.pdf",
                    "location": "page 8",
                },
                "distance": 0.1,
            }
        ]

    def run_text_case(self, question: str) -> None:
        """Check that a normal question calls the text answerer, not image APIs."""

        fake_client = Mock()
        with (
            patch.object(chatbot, "retrieve_relevant_chunks", return_value=self.matches) as retrieve,
            patch.object(chatbot, "get_openai_client", return_value=fake_client),
            patch.object(chatbot, "answer_with_rag", return_value="Text answer") as answer_text,
            patch.object(chatbot, "build_image_prompt") as build_prompt,
            patch.object(chatbot, "generate_image") as generate_image,
        ):
            result = chatbot.process_user_question(
                question, self.collection, Mock(), [], self.config
            )

        retrieve.assert_called_once()
        answer_text.assert_called_once()
        build_prompt.assert_not_called()
        generate_image.assert_not_called()
        self.assertEqual(result["answer"], "Text answer")
        self.assertIsNone(result["image_bytes"])
        self.assertFalse(result["is_error"])

    def test_revenue_question_returns_text_only(self) -> None:
        """A revenue question uses retrieval and normal text generation."""

        self.run_text_case("What was revenue in FY2025?")

    def test_presentation_summary_returns_text_only(self) -> None:
        """A normal presentation summary does not invoke image generation."""

        self.run_text_case("Summarize the investor presentation")

    def test_hypothetical_infographic_question_returns_text_only(self) -> None:
        """Asking what a visual could contain is not a request to create one."""

        self.run_text_case("What information would a revenue infographic contain?")

    def test_tentative_visual_request_asks_for_clarification(self) -> None:
        """Uncertain wording should not trigger image generation unexpectedly."""

        with (
            patch.object(chatbot, "retrieve_relevant_chunks") as retrieve,
            patch.object(chatbot, "get_openai_client") as get_client,
        ):
            result = chatbot.process_user_question(
                "Maybe we should create an infographic showing revenue",
                self.collection,
                Mock(),
                [],
                self.config,
            )

        self.assertIn("Would you like me to generate an image", result["answer"])
        retrieve.assert_not_called()
        get_client.assert_not_called()

    def test_direct_visualize_verb_is_explicit_image_intent(self) -> None:
        """A direct request to visualize information need not say 'image'."""

        self.assertTrue(
            chatbot.is_image_generation_request("Visualize the company's business segments")
        )

    def run_image_case(self, question: str) -> None:
        """Check that an explicit visual request reuses retrieved evidence."""

        fake_client = Mock()
        brief = chatbot.InfographicBrief(
            image_prompt="Portrait financial visual background without text.",
            facts=[
                {
                    "text": "Revenue was INR 100 crore in FY2025.",
                    "source": "annual-report.pdf",
                    "location": "page 8",
                }
            ],
        )
        with (
            patch.object(chatbot, "retrieve_relevant_chunks", return_value=self.matches) as retrieve,
            patch.object(chatbot, "get_openai_client", return_value=fake_client),
            patch.object(chatbot, "build_image_prompt", return_value=brief) as build_prompt,
            patch.object(chatbot, "generate_image", return_value=b"raw-image") as generate_image,
            patch.object(chatbot, "render_verified_infographic", return_value=b"png-image") as render_image,
            patch.object(chatbot, "answer_with_rag") as answer_text,
        ):
            result = chatbot.process_user_question(
                question, self.collection, Mock(), [], self.config
            )

        retrieve.assert_called_once()
        build_prompt.assert_called_once_with(question, self.matches, self.config, fake_client)
        generate_image.assert_called_once_with(brief.image_prompt, self.config, fake_client)
        render_image.assert_called_once_with(b"raw-image", brief.facts)
        answer_text.assert_not_called()
        self.assertEqual(result["image_bytes"], b"png-image")
        self.assertEqual(result["sources"][0]["label"], "annual-report.pdf (page 8)")
        self.assertFalse(result["is_error"])

    def test_a4_revenue_infographic_uses_retrieved_data(self) -> None:
        """An A4-style infographic request retrieves its evidence before image creation."""

        self.run_image_case("Create an A4 infographic showing revenue growth")

    def test_business_segments_infographic_uses_retrieved_data(self) -> None:
        """A business-segments visual follows the same shared RAG retrieval path."""

        self.run_image_case("Generate an infographic about the company's business segments")

    def test_graph_request_draws_verified_chart_without_image_model(self) -> None:
        """A graph is rendered locally from verified points, not image-model pixels."""

        chart_points = [
            {
                "period": "FY2024",
                "value": 10.0,
                "display_value": "10%",
                "source": "annual-report.pdf",
                "location": "page 8",
            },
            {
                "period": "FY2025",
                "value": 12.0,
                "display_value": "12%",
                "source": "annual-report.pdf",
                "location": "page 9",
            },
        ]
        brief = chatbot.InfographicBrief(
            image_prompt="Not used for a chart.",
            facts=[],
            chart_title="Revenue growth",
            chart_points=chart_points,
        )
        fake_client = Mock()
        question = "Create a graph for revenue growth"
        with (
            patch.object(chatbot, "retrieve_relevant_chunks", return_value=self.matches),
            patch.object(chatbot, "get_openai_client", return_value=fake_client),
            patch.object(chatbot, "build_image_prompt", return_value=brief),
            patch.object(chatbot, "render_verified_chart", return_value=b"chart-png") as render_chart,
            patch.object(chatbot, "generate_image") as generate_background,
        ):
            result = chatbot.process_user_question(
                question, self.collection, Mock(), [], self.config
            )

        render_chart.assert_called_once_with(chart_points, "Revenue growth", "1024x1536")
        generate_background.assert_not_called()
        self.assertEqual(result["image_bytes"], b"chart-png")
        self.assertFalse(result["is_error"])

    def test_graph_request_refuses_to_invent_missing_data(self) -> None:
        """A single verified point is insufficient to draw a trend graph."""

        brief = chatbot.InfographicBrief(
            image_prompt="Unused for a graph.",
            facts=[],
            chart_title="Revenue growth",
            chart_points=[],
        )
        with (
            patch.object(chatbot, "retrieve_relevant_chunks", return_value=self.matches),
            patch.object(chatbot, "get_openai_client", return_value=Mock()),
            patch.object(chatbot, "build_image_prompt", return_value=brief),
            patch.object(chatbot, "render_verified_chart") as render_chart,
            patch.object(chatbot, "generate_image") as generate_background,
        ):
            result = chatbot.process_user_question(
                "Create a graph for revenue growth",
                self.collection,
                Mock(),
                [],
                self.config,
            )

        self.assertIn("will not invent missing data", result["answer"])
        self.assertIsNone(result["image_bytes"])
        render_chart.assert_not_called()
        generate_background.assert_not_called()

    def test_missing_key_and_image_failure_return_helpful_errors(self) -> None:
        """Missing credentials and image endpoint failures become UI-safe results."""

        question = "Create an infographic showing revenue growth"
        with patch.object(chatbot, "retrieve_relevant_chunks", return_value=self.matches):
            missing_key_config = chatbot.AppConfig(
                **{**self.config.__dict__, "openai_api_key": ""}
            )
            missing_key_result = chatbot.process_user_question(
                question, self.collection, Mock(), [], missing_key_config
            )

        self.assertTrue(missing_key_result["is_error"])
        self.assertIn("OPENAI_API_KEY", missing_key_result["answer"])

        fake_client = Mock()
        connection_error = APIConnectionError(request=Mock())
        brief = chatbot.InfographicBrief(
            image_prompt="Portrait background without text.",
            facts=[],
        )
        with (
            patch.object(chatbot, "retrieve_relevant_chunks", return_value=self.matches),
            patch.object(chatbot, "get_openai_client", return_value=fake_client),
            patch.object(chatbot, "build_image_prompt", return_value=brief),
            patch.object(chatbot, "generate_image", side_effect=connection_error),
        ):
            failed_image_result = chatbot.process_user_question(
                question, self.collection, Mock(), [], self.config
            )

        self.assertTrue(failed_image_result["is_error"])
        self.assertIn("connect to OpenAI", failed_image_result["answer"])
        self.assertIsNone(failed_image_result["image_bytes"])

    def test_image_brief_discards_facts_not_quoted_in_retrieved_context(self) -> None:
        """Only exact source-supported quotations are accepted for the infographic."""

        response = Mock()
        response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "image_prompt": "Portrait abstract financial background.",
                            "facts": [
                                {
                                    "text": "Revenue was INR 100 crore in FY2025.",
                                    "source_index": 0,
                                },
                                {
                                    "text": "Revenue was INR 999 crore in FY2025.",
                                    "source_index": 0,
                                },
                            ],
                        }
                    )
                )
            )
        ]
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = response

        brief = chatbot.build_image_prompt(
            "Create a revenue infographic", self.matches, self.config, fake_client
        )

        self.assertEqual(len(brief.facts), 1)
        self.assertEqual(brief.facts[0]["text"], self.matches[0]["text"])
        fake_client.chat.completions.create.assert_called_once()

    def test_image_brief_keeps_only_source_verified_chart_points(self) -> None:
        """Chart periods, metric, exact values, and quotations must match sources."""

        chart_matches = [
            {
                "text": "Revenue growth was 10% in FY2024.",
                "metadata": {"source": "annual-report.pdf", "location": "page 8"},
            },
            {
                "text": "Revenue growth was 12% in FY2025.",
                "metadata": {"source": "annual-report.pdf", "location": "page 9"},
            },
        ]
        response = Mock()
        response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "image_prompt": "Plain background.",
                            "facts": [],
                            "chart_title": "Revenue growth",
                            "chart_series": "Revenue growth",
                            "chart_points": [
                                {
                                    "period": "FY2024",
                                    "value": 10,
                                    "display_value": "10%",
                                    "quote": chart_matches[0]["text"],
                                    "source_index": 0,
                                },
                                {
                                    "period": "FY2025",
                                    "value": 12,
                                    "display_value": "12%",
                                    "quote": chart_matches[1]["text"],
                                    "source_index": 1,
                                },
                                {
                                    "period": "FY2026",
                                    "value": 99,
                                    "display_value": "99%",
                                    "quote": "Revenue growth was 99% in FY2026.",
                                    "source_index": 1,
                                },
                                {
                                    "period": "FY2021-FY2025",
                                    "value": 10.2,
                                    "display_value": "10.2%",
                                    "quote": "Revenue growth CAGR was 10.2% over FY2021-FY2025.",
                                    "source_index": 1,
                                },
                            ],
                        }
                    )
                )
            )
        ]
        fake_client = Mock()
        fake_client.chat.completions.create.return_value = response

        brief = chatbot.build_image_prompt(
            "Create a revenue growth graph", chart_matches, self.config, fake_client
        )

        self.assertEqual(len(brief.chart_points or []), 2)
        self.assertEqual(
            [point["period"] for point in brief.chart_points or []],
            ["FY2024", "FY2025"],
        )

    def test_infographic_renderer_returns_a_valid_png(self) -> None:
        """The final composed image can be opened as a PNG at the original size."""

        source = Image.new("RGB", (512, 768), color="#4c626b")
        source_buffer = BytesIO()
        source.save(source_buffer, format="PNG")

        image_bytes = chatbot.render_verified_infographic(
            source_buffer.getvalue(),
            [
                {
                    "text": "Revenue was INR 100 crore in FY2025.",
                    "source": "annual-report.pdf",
                    "location": "page 8",
                }
            ],
        )

        with Image.open(BytesIO(image_bytes)) as rendered_image:
            self.assertEqual(rendered_image.format, "PNG")
            self.assertEqual(rendered_image.size, (512, 768))

    def test_image_generation_uses_configured_supported_options(self) -> None:
        """The SDK receives the configured model, portrait size, and PNG format."""

        encoded_png = base64.b64encode(b"png-test-bytes").decode("ascii")
        fake_client = Mock()
        fake_client.images.generate.return_value = Mock(
            data=[Mock(b64_json=encoded_png)]
        )

        image_bytes = chatbot.generate_image(
            "Portrait background without text.", self.config, fake_client
        )

        self.assertEqual(image_bytes, b"png-test-bytes")
        fake_client.images.generate.assert_called_once_with(
            model="gpt-image-2.5-flare",
            prompt="Portrait background without text.",
            size="1024x1536",
            quality="medium",
            output_format="png",
        )

    def test_verified_chart_renderer_returns_png_with_configured_size(self) -> None:
        """A local chart image preserves the configured portrait dimensions."""

        image_bytes = chatbot.render_verified_chart(
            [
                {
                    "period": "FY2024",
                    "value": 10,
                    "display_value": "10%",
                    "source": "annual-report.pdf",
                    "location": "page 8",
                },
                {
                    "period": "FY2025",
                    "value": 12,
                    "display_value": "12%",
                    "source": "annual-report.pdf",
                    "location": "page 9",
                },
            ],
            "Revenue growth",
            "512x768",
        )

        with Image.open(BytesIO(image_bytes)) as chart_image:
            self.assertEqual(chart_image.format, "PNG")
            self.assertEqual(chart_image.size, (512, 768))


if __name__ == "__main__":
    unittest.main()