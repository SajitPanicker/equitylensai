"""Offline tests for metadata indexing, filtered retrieval, and recursive chunking."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stock_rag_chatbot as chatbot


class MemoryCollection:
    """Small Chroma-shaped store used to test filters and safe upserts without a database."""

    def __init__(self):
        self.records = {}

    def upsert(self, ids, documents, embeddings, metadatas):
        for record_id, document, metadata in zip(ids, documents, metadatas):
            self.records[record_id] = {"document": document, "metadata": metadata}

    def get(self, where=None, include=None):
        selected = [
            (record_id, record)
            for record_id, record in self.records.items()
            if self._matches(record["metadata"], where)
        ]
        return {
            "ids": [record_id for record_id, _ in selected],
            "documents": [record["document"] for _, record in selected],
            "metadatas": [record["metadata"] for _, record in selected],
        }

    def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)

    def count(self):
        return len(self.records)

    def query(self, query_embeddings, n_results, include, where=None, where_document=None):
        selected = [
            record
            for record in self.records.values()
            if self._matches(record["metadata"], where)
            and self._matches_document(record["document"], where_document)
        ][:n_results]
        return {
            "documents": [[record["document"] for record in selected]],
            "metadatas": [[record["metadata"] for record in selected]],
            "distances": [[0.1 for _ in selected]],
        }

    @classmethod
    def _matches(cls, metadata, where):
        if not where:
            return True
        if "$and" in where:
            return all(cls._matches(metadata, item) for item in where["$and"])
        if "$or" in where:
            return any(cls._matches(metadata, item) for item in where["$or"])
        return all(
            metadata.get(key) in value["$in"]
            if isinstance(value, dict) and "$in" in value
            else metadata.get(key) == value
            for key, value in where.items()
        )

    @classmethod
    def _matches_document(cls, document, where):
        if not where:
            return True
        if "$and" in where:
            return all(cls._matches_document(document, item) for item in where["$and"])
        if "$or" in where:
            return any(cls._matches_document(document, item) for item in where["$or"])
        if "$contains" in where:
            return where["$contains"].casefold() in document.casefold()
        return True


class MetadataIndexingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary_directory.name)
        self.collection = MemoryCollection()
        self.model = object()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def add_document(self, filename, content="revenue growth 12 percent"):
        path = self.folder / filename
        path.write_text(content, encoding="utf-8")
        return path

    def index(self, file_metadata=None):
        def extract(path, progress_callback=None):
            return [chatbot.DocumentSegment(path.read_text(encoding="utf-8"), "page 12")]

        with (
            patch.object(chatbot, "extract_document_text", side_effect=extract),
            patch.object(
                chatbot,
                "create_embeddings",
                side_effect=lambda texts, model, progress_callback=None: [
                    [0.0] * chatbot.EXPECTED_EMBEDDING_DIMENSION for _ in texts
                ],
            ),
        ):
            return chatbot.index_documents(
                self.folder,
                self.collection,
                self.model,
                1000,
                150,
                lambda fraction, message: None,
                file_metadata=file_metadata,
            )

    def metadata_for(self, path, company, ticker, year, quarter="Q1"):
        return {
            str(path.resolve()): {
                "company": company,
                "ticker": ticker,
                "fiscal_year": year,
                "quarter": quarter,
                "document_type": "Investor Presentation",
            }
        }

    def retrieve(self, question, filters):
        with patch.object(
            chatbot,
            "create_embeddings",
            return_value=[[0.0] * chatbot.EXPECTED_EMBEDDING_DIMENSION],
        ):
            return chatbot.retrieve_relevant_chunks(
                question, self.collection, self.model, 10, filters=filters
            )

    def test_two_years_and_two_companies_remain_separate(self):
        fy2025 = self.add_document("Mahindra and Mahindra__M&M__FY2025__Q1__Investor_Presentation.pdf")
        fy2026 = self.add_document("Mahindra and Mahindra__M&M__FY2026__Q1__Investor_Presentation.pdf")
        tata = self.add_document("Tata Motors__TATAMOTORS__FY2026__Q1__Investor_Presentation.pptx")

        report = self.index()

        self.assertEqual(report["successful_documents"], 3)
        metadata = [record["metadata"] for record in self.collection.records.values()]
        self.assertEqual({item["company"] for item in metadata}, {"Mahindra and Mahindra", "Tata Motors"})
        self.assertEqual({item["fiscal_year"] for item in metadata}, {"FY2025", "FY2026"})
        self.assertNotEqual(
            next(item for item in metadata if item["source"] == fy2025.name)["source_path"],
            next(item for item in metadata if item["source"] == fy2026.name)["source_path"],
        )
        self.assertTrue(any(item["source"] == tata.name for item in metadata))

    def test_company_year_quarter_filter_returns_only_requested_evidence(self):
        self.add_document("Mahindra and Mahindra__M&M__FY2025__Q1__Investor_Presentation.pdf")
        self.add_document("Mahindra and Mahindra__M&M__FY2026__Q1__Investor_Presentation.pdf")
        self.add_document("Tata Motors__TATAMOTORS__FY2026__Q1__Investor_Presentation.pdf")
        self.index()

        matches = self.retrieve(
            "revenue growth",
            {"company": "Mahindra and Mahindra", "fiscal_year": "FY2026", "quarter": "Q1"},
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["metadata"]["ticker"], "M&M")
        self.assertEqual(matches[0]["metadata"]["location"], "page 12")

    def test_multi_year_comparison_queries_years_separately(self):
        self.add_document("Mahindra and Mahindra__M&M__FY2025__Q1__Investor_Presentation.pdf")
        self.add_document("Mahindra and Mahindra__M&M__FY2026__Q1__Investor_Presentation.pdf")
        self.index()

        matches = self.retrieve(
            "compare revenue growth",
            {"company": "Mahindra and Mahindra", "fiscal_year": ["FY2025", "FY2026"]},
        )

        self.assertEqual({item["metadata"]["fiscal_year"] for item in matches}, {"FY2025", "FY2026"})

    def test_compare_two_companies_uses_valid_or_filter(self):
        self.add_document("Mahindra and Mahindra__M&M__FY2026__Q1__Investor_Presentation.pdf")
        self.add_document("Tata Motors__TATAMOTORS__FY2026__Q1__Investor_Presentation.pdf")
        self.index()
        filters = chatbot.resolve_question_filters(
            "Compare Mahindra and Mahindra and Tata Motors",
            chatbot.get_indexed_document_metadata(self.collection),
        )

        matches = self.retrieve("compare", filters)

        self.assertEqual(
            {item["metadata"]["company"] for item in matches},
            {"Mahindra and Mahindra", "Tata Motors"},
        )
        self.assertEqual(
            chatbot.build_metadata_where({"company": ["Mahindra and Mahindra", "Tata Motors"]}),
            {"company": {"$in": ["Mahindra and Mahindra", "Tata Motors"]}},
        )

    def test_question_filters_recognize_bare_year_and_ticker_comparison(self):
        metadata = [
            {"company": "Infosys", "ticker": "INFY", "fiscal_year": "FY2026"},
            {
                "company": "Tata Consultancy Services",
                "ticker": "TCS",
                "fiscal_year": "FY2026",
            },
        ]

        filters = chatbot.resolve_question_filters(
            "Can you compare Infosys 2026 and TCS 2026 results", metadata
        )

        self.assertEqual(
            filters,
            {
                "company": ["Infosys", "Tata Consultancy Services"],
                "fiscal_year": ["FY2026"],
            },
        )

    def test_reindexing_same_file_does_not_duplicate_chunks(self):
        self.add_document("Mahindra and Mahindra__M&M__FY2026__Q1__Investor_Presentation.pdf")
        self.index()
        initial_ids = set(self.collection.records)
        self.index()

        self.assertEqual(set(self.collection.records), initial_ids)
        self.assertEqual(len(self.collection.records), 1)

    def test_changed_content_replaces_old_chunks(self):
        path = self.add_document("Mahindra and Mahindra__M&M__FY2026__Q1__Investor_Presentation.pdf", "old revenue 10")
        self.index()
        old_ids = set(self.collection.records)
        path.write_text("new revenue 20", encoding="utf-8")
        self.index()

        self.assertTrue(old_ids.isdisjoint(self.collection.records))
        self.assertEqual([item["document"] for item in self.collection.records.values()], ["new revenue 20"])

    def test_metadata_only_change_updates_existing_records(self):
        path = self.add_document("presentation.pdf")
        original = self.metadata_for(path, "Example Company", "EX", "FY2025")
        updated = self.metadata_for(path, "Example Company", "EX", "FY2026")
        self.index(original)
        ids_before = set(self.collection.records)
        self.index(updated)

        self.assertEqual(set(self.collection.records), ids_before)
        self.assertEqual(
            {item["metadata"]["fiscal_year"] for item in self.collection.records.values()},
            {"FY2026"},
        )

    def test_missing_requested_company_year_reports_no_match(self):
        path = self.add_document("Example Company__EX__FY2025__Q1__Investor_Presentation.pdf")
        self.index()
        config = chatbot.AppConfig(
            openai_api_key="", documents_folder=self.folder, chroma_db_path=self.folder,
            embedding_model="sentence-transformers/all-mpnet-base-v2", openai_model="gpt-4o-mini",
            image_model="image-model", image_size="1024x1536", image_quality="medium",
            top_k=4, chunk_size=1000, chunk_overlap=150,
        )

        result = chatbot.process_user_question(
            "Revenue?", self.collection, self.model, [], config,
            filters={"company": "Example Company", "fiscal_year": "FY2026"},
        )

        self.assertIn("No indexed documents match", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_legacy_record_is_replaced_during_safe_migration(self):
        path = self.add_document("presentation.pdf")
        file_hash = chatbot.get_file_hash(path)
        legacy_id = f"legacy-{file_hash[:16]}-1"
        self.collection.records[legacy_id] = {
            "document": "legacy revenue evidence",
            "metadata": {
                "source": path.name,
                "source_path": str(path.resolve()),
                "file_hash": file_hash,
                "chunk_number": 1,
                "location": "page 12",
            },
        }

        self.index(self.metadata_for(path, "Example Company", "EX", "FY2026"))

        self.assertNotIn(legacy_id, self.collection.records)
        migrated_metadata = next(iter(self.collection.records.values()))["metadata"]
        self.assertEqual(migrated_metadata["company"], "Example Company")
        self.assertEqual(migrated_metadata["fiscal_year"], "FY2026")

    def test_where_document_restricts_results_to_matching_text(self):
        self.add_document(
            "Mahindra and Mahindra__M&M__FY2026__Q1__Investor_Presentation.pdf",
            "Revenue growth was twelve percent.",
        )
        self.add_document(
            "Tata Motors__TATAMOTORS__FY2026__Q1__Investor_Presentation.pdf",
            "Vehicle volumes were stable.",
        )
        self.index()
        with patch.object(
            chatbot,
            "create_embeddings",
            return_value=[[0.0] * chatbot.EXPECTED_EMBEDDING_DIMENSION],
        ):
            matches = chatbot.retrieve_relevant_chunks(
                "financial evidence",
                self.collection,
                self.model,
                10,
                where_document="revenue",
            )

        self.assertEqual(len(matches), 1)
        self.assertIn("Revenue", matches[0]["text"])

    def test_incomplete_requested_years_refuse_chart_generation(self):
        path = self.add_document(
            "Example Company__EX__FY2026__Q1__Investor_Presentation.pdf"
        )
        self.index(self.metadata_for(path, "Example Company", "EX", "FY2026"))
        config = chatbot.AppConfig(
            openai_api_key="", documents_folder=self.folder, chroma_db_path=self.folder,
            embedding_model="sentence-transformers/all-mpnet-base-v2", openai_model="gpt-4o-mini",
            image_model="image-model", image_size="1024x1536", image_quality="medium",
            top_k=4, chunk_size=1000, chunk_overlap=150,
        )

        with (
            patch.object(chatbot, "get_openai_client") as get_client,
            patch.object(
                chatbot,
                "create_embeddings",
                return_value=[[0.0] * chatbot.EXPECTED_EMBEDDING_DIMENSION],
            ),
        ):
            result = chatbot.process_user_question(
                "Create a chart for revenue growth",
                self.collection,
                self.model,
                [],
                config,
                filters={"company": "Example Company", "fiscal_year": ["FY2025", "FY2026"]},
            )

        self.assertIn("incomplete requested comparison", result["answer"])
        self.assertIsNone(result["image_bytes"])
        get_client.assert_not_called()

    def test_recursive_chunks_keep_page_and_slide_locations_separate(self):
        chunks = chatbot.chunk_text_segments(
            [
                chatbot.DocumentSegment("Paragraph one. " * 20, "page 5"),
                chatbot.DocumentSegment("Slide two. " * 20, "slide 2"),
                chatbot.DocumentSegment("   ", "page 6"),
            ],
            chunk_size=45,
            chunk_overlap=8,
        )

        self.assertTrue(all(len(chunk.text) <= 45 for chunk in chunks))
        self.assertTrue(all(chunk.chunk_number == index for index, chunk in enumerate(chunks, 1)))
        self.assertIn("page 5", {chunk.location for chunk in chunks})
        self.assertIn("slide 2", {chunk.location for chunk in chunks})
        self.assertNotIn("page 6", {chunk.location for chunk in chunks})
        self.assertTrue(all(not ("Paragraph one." in chunk.text and "Slide two." in chunk.text) for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
