"""
Tests unitaires — Indexation FAISS
Vérifie le chargement de l'index et les recherches de similarité.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_doc():
    return Document(
        page_content="Concert de jazz au Pan Piper",
        metadata={
            "uid": "abc123",
            "title": "PEGAZZ & l'HELICON : Rose(S)",
            "firstdate_begin": "2026-01-17T19:00:00+00:00",
            "lastdate_end": "2026-01-17T21:00:00+00:00",
            "location_name": "Le Pan Piper",
            "location_address": "2 Impasse Lamier",
            "city": "Paris",
            "canonicalurl": "https://openagenda.com/culture/events/pegazz-and-lhelicon-roses",
        },
    )


@pytest.fixture
def fake_docs(fake_doc):
    doc2 = Document(
        page_content="Projection organisée par le Parlement européen",
        metadata={
            "uid": "def456",
            "title": "Prix Lux 2026 : avant-première du film Sorda",
            "firstdate_begin": "2026-03-05T18:30:00+00:00",
            "lastdate_end": "2026-03-05T21:30:00+00:00",
            "location_name": "Cinéma des Cinéastes",
            "location_address": "7 Avenue de Clichy",
            "city": "Paris",
            "canonicalurl": "https://openagenda.com/europe-reunion/events/prix-lux-2026",
        },
    )
    return [fake_doc, doc2]


@pytest.fixture
def mock_vectorstore(fake_docs):
    """FAISS vectorstore simulé sans charger de vrais embeddings."""
    vs = MagicMock()
    vs.similarity_search.return_value = fake_docs
    vs.similarity_search_with_relevance_scores.return_value = [
        (fake_docs[0], 0.85),
        (fake_docs[1], 0.72),
    ]
    return vs


# ── Tests chargement ──────────────────────────────────────────────────────────

class TestFAISSLoading:
    def test_load_local_called_with_correct_path(self):
        """FAISS.load_local doit être appelé avec le bon répertoire et allow_dangerous_deserialization=True."""
        with (
            patch("rag.engine.FAISS.load_local") as mock_load,
            patch("rag.engine.HuggingFaceEmbeddings") as mock_emb,
            patch("rag.engine.ChatMistralAI"),
        ):
            mock_load.return_value = MagicMock()
            mock_emb.return_value = MagicMock()

            from rag.engine import RAGEngine, RAGConfig
            RAGEngine(RAGConfig(faiss_dir="vectorstores/faiss_index"))

            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args
            assert call_kwargs.kwargs.get("allow_dangerous_deserialization") is True

    def test_embedding_model_name(self):
        """Le bon modèle d'embeddings doit être utilisé."""
        with (
            patch("rag.engine.FAISS.load_local"),
            patch("rag.engine.HuggingFaceEmbeddings") as mock_emb,
            patch("rag.engine.ChatMistralAI"),
        ):
            mock_emb.return_value = MagicMock()

            from rag.engine import RAGEngine, RAGConfig
            cfg = RAGConfig(embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            RAGEngine(cfg)

            mock_emb.assert_called_once_with(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                encode_kwargs={"normalize_embeddings": True},
            )


# ── Tests similarity_search ───────────────────────────────────────────────────

class TestSimilaritySearch:
    def test_similarity_search_returns_documents(self, mock_vectorstore, fake_docs):
        results = mock_vectorstore.similarity_search("concert jazz Paris", k=10)
        assert len(results) == len(fake_docs)
        assert all(isinstance(d, Document) for d in results)

    def test_similarity_search_with_scores_returns_tuples(self, mock_vectorstore):
        results = mock_vectorstore.similarity_search_with_relevance_scores("jazz", k=10)
        assert len(results) == 2
        for doc, score in results:
            assert isinstance(doc, Document)
            assert 0.0 <= score <= 1.0

    def test_score_threshold_filters_low_scores(self, mock_vectorstore):
        """Les documents sous le seuil de score doivent être écartés."""
        mock_vectorstore.similarity_search_with_relevance_scores.return_value = [
            (Document(page_content="hors sujet", metadata={}), 0.15),
            (Document(page_content="pertinent", metadata={}), 0.80),
        ]
        results = mock_vectorstore.similarity_search_with_relevance_scores("jazz", k=10)
        threshold = 0.3
        filtered = [doc for doc, score in results if score >= threshold]
        assert len(filtered) == 1
        assert filtered[0].page_content == "pertinent"


# ── Tests métadonnées ─────────────────────────────────────────────────────────

class TestDocumentMetadata:
    expected_fields = [
        "uid", "title", "firstdate_begin", "lastdate_end",
        "location_name", "location_address", "city", "canonicalurl",
    ]

    def test_document_has_expected_metadata_fields(self, fake_doc):
        for field in self.expected_fields:
            assert field in fake_doc.metadata, f"Champ manquant : {field}"

    def test_document_page_content_not_empty(self, fake_doc):
        assert fake_doc.page_content.strip() != ""

    def test_uid_is_string(self, fake_doc):
        uid = fake_doc.metadata.get("uid")
        assert isinstance(uid, str)
