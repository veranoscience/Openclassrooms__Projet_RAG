"""
Tests unitaires — RAGEngine
Couvre : parse_date, overlaps, deduplicate_docs, sort_by_start_date,
         _retrieve, _filter, _build_context, ask
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document

from rag.engine import (
    parse_date,
    overlaps,
    deduplicate_docs,
    sort_by_start_date,
    RAGConfig,
    RAGEngine,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_doc(uid, title, begin, end, page_content="description"):
    return Document(
        page_content=page_content,
        metadata={
            "uid": uid,
            "title": title,
            "firstdate_begin": begin,
            "lastdate_end": end,
            "location_name": "Salle Test",
            "location_address": "1 rue Test",
            "city": "Paris",
            "canonicalurl": f"https://example.com/{uid}",
        },
    )


# ── Fixture : engine avec mocks ───────────────────────────────────────────────

@pytest.fixture
def engine():
    with (
        patch("langchain_community.vectorstores.FAISS.load_local") as mock_faiss,
        patch("langchain_huggingface.HuggingFaceEmbeddings"),
        patch("langchain_mistralai.ChatMistralAI") as mock_llm,
    ):
        mock_faiss.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        eng = RAGEngine(RAGConfig())
        eng.vs = MagicMock()
        eng.llm = MagicMock()
        yield eng


# ── Tests parse_date ──────────────────────────────────────────────────────────

class TestParseDate:
    def test_full_date(self):
        result = parse_date("événement le 5 mars 2026")
        assert result is not None
        start, end = result
        assert start == pd.Timestamp(2026, 3, 5, tz="UTC")
        assert end.date() == pd.Timestamp(2026, 3, 5, tz="UTC").date()

    def test_month_year(self):
        result = parse_date("concerts jazz à Paris en janvier 2026")
        assert result is not None
        start, end = result
        assert start == pd.Timestamp(2026, 1, 1, tz="UTC")
        assert end == pd.Timestamp(2026, 1, 31, 23, 59, 59, tz="UTC")

    def test_year_only(self):
        result = parse_date("événements à Paris en 2026")
        assert result is not None
        start, end = result
        assert start == pd.Timestamp(2026, 1, 1, tz="UTC")
        assert end == pd.Timestamp(2026, 12, 31, 23, 59, 59, tz="UTC")

    def test_no_date(self):
        assert parse_date("événements culturels à Paris") is None

    def test_premier_jour(self):
        result = parse_date("le 1er avril 2026")
        assert result is not None
        start, _ = result
        assert start.day == 1
        assert start.month == 4

    def test_accent_fevrier(self):
        result = parse_date("en février 2026")
        assert result is not None
        start, end = result
        assert start.month == 2
        assert end.month == 2

    def test_accent_decembre(self):
        result = parse_date("en décembre 2026")
        assert result is not None
        start, _ = result
        assert start.month == 12


# ── Tests overlaps ────────────────────────────────────────────────────────────

class TestOverlaps:
    def test_event_inside_range(self):
        doc = make_doc("1", "Test", "2026-03-05T18:00:00+00:00", "2026-03-05T21:00:00+00:00")
        start = pd.Timestamp(2026, 3, 1, tz="UTC")
        end = pd.Timestamp(2026, 3, 31, 23, 59, 59, tz="UTC")
        assert overlaps(doc, start, end) is True

    def test_event_before_range(self):
        doc = make_doc("1", "Test", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
        start = pd.Timestamp(2026, 3, 1, tz="UTC")
        end = pd.Timestamp(2026, 3, 31, 23, 59, 59, tz="UTC")
        assert overlaps(doc, start, end) is False

    def test_event_after_range(self):
        doc = make_doc("1", "Test", "2026-05-01T00:00:00+00:00", "2026-05-31T23:59:59+00:00")
        start = pd.Timestamp(2026, 3, 1, tz="UTC")
        end = pd.Timestamp(2026, 3, 31, 23, 59, 59, tz="UTC")
        assert overlaps(doc, start, end) is False

    def test_event_spans_range(self):
        """Un événement qui chevauche la période doit être inclus."""
        doc = make_doc("1", "Test", "2026-02-01T00:00:00+00:00", "2026-04-30T23:59:59+00:00")
        start = pd.Timestamp(2026, 3, 1, tz="UTC")
        end = pd.Timestamp(2026, 3, 31, 23, 59, 59, tz="UTC")
        assert overlaps(doc, start, end) is True

    def test_missing_dates_returns_false(self):
        doc = Document(page_content="", metadata={"uid": "x", "title": "X"})
        start = pd.Timestamp(2026, 3, 1, tz="UTC")
        end = pd.Timestamp(2026, 3, 31, 23, 59, 59, tz="UTC")
        assert overlaps(doc, start, end) is False


# ── Tests deduplicate_docs ────────────────────────────────────────────────────

class TestDeduplicateDocs:
    def test_removes_duplicates(self):
        doc = make_doc("uid1", "Concert", "2026-01-01T00:00:00+00:00", "2026-01-01T23:59:59+00:00")
        docs = [doc, doc]
        result = deduplicate_docs(docs)
        assert len(result) == 1

    def test_keeps_all_unique(self):
        docs = [
            make_doc("uid1", "A", "2026-01-01T00:00:00+00:00", "2026-01-01T23:59:59+00:00"),
            make_doc("uid2", "B", "2026-01-02T00:00:00+00:00", "2026-01-02T23:59:59+00:00"),
        ]
        result = deduplicate_docs(docs)
        assert len(result) == 2

    def test_preserves_first_occurrence(self):
        doc1 = make_doc("uid1", "Premier", "2026-01-01T00:00:00+00:00", "2026-01-01T23:59:59+00:00")
        doc2 = make_doc("uid1", "Doublon", "2026-01-01T00:00:00+00:00", "2026-01-01T23:59:59+00:00")
        result = deduplicate_docs([doc1, doc2])
        assert result[0].metadata["title"] == "Premier"


# ── Tests sort_by_start_date ──────────────────────────────────────────────────

class TestSortByStartDate:
    def test_sorts_ascending(self):
        docs = [
            make_doc("3", "C", "2026-03-01T00:00:00+00:00", "2026-03-31T23:59:59+00:00"),
            make_doc("1", "A", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00"),
            make_doc("2", "B", "2026-02-01T00:00:00+00:00", "2026-02-28T23:59:59+00:00"),
        ]
        result = sort_by_start_date(docs)
        titles = [d.metadata["title"] for d in result]
        assert titles == ["A", "B", "C"]


# ── Tests _retrieve ───────────────────────────────────────────────────────────

class TestRetrieve:
    def test_filters_by_score_threshold(self, engine):
        doc_high = make_doc("1", "Pertinent", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
        doc_low = make_doc("2", "Hors sujet", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
        engine.vs.similarity_search_with_relevance_scores.return_value = [
            (doc_high, 0.85),
            (doc_low, 0.10),
        ]
        engine.config.score_threshold = 0.3
        result = engine._retrieve("concert jazz Paris")
        assert len(result) == 1
        assert result[0].metadata["title"] == "Pertinent"

    def test_returns_all_above_threshold(self, engine):
        docs = [make_doc(str(i), f"Event {i}", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00") for i in range(3)]
        engine.vs.similarity_search_with_relevance_scores.return_value = [
            (docs[0], 0.90), (docs[1], 0.50), (docs[2], 0.20),
        ]
        engine.config.score_threshold = 0.3
        result = engine._retrieve("jazz")
        assert len(result) == 2

    def test_empty_result_when_all_below_threshold(self, engine):
        doc = make_doc("1", "Peu pertinent", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
        engine.vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.05)]
        engine.config.score_threshold = 0.3
        result = engine._retrieve("question sans rapport")
        assert result == []


# ── Tests _filter ─────────────────────────────────────────────────────────────

class TestFilter:
    def test_temporal_filter_keeps_matching(self, engine):
        doc_march = make_doc("1", "Mars", "2026-03-01T00:00:00+00:00", "2026-03-31T23:59:59+00:00")
        doc_jan = make_doc("2", "Janvier", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
        result = engine._filter([doc_march, doc_jan], "événements en mars 2026")
        titles = [d.metadata["title"] for d in result]
        assert "Mars" in titles
        assert "Janvier" not in titles

    def test_no_date_keeps_all(self, engine):
        docs = [
            make_doc("1", "A", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00"),
            make_doc("2", "B", "2026-05-01T00:00:00+00:00", "2026-05-31T23:59:59+00:00"),
        ]
        result = engine._filter(docs, "événements culturels à Paris")
        assert len(result) == 2

    def test_deduplication_applied(self, engine):
        doc = make_doc("uid1", "Concert", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
        result = engine._filter([doc, doc], "événements culturels à Paris")
        assert len(result) == 1

    def test_result_sorted_by_date(self, engine):
        docs = [
            make_doc("3", "C", "2026-03-01T00:00:00+00:00", "2026-12-31T23:59:59+00:00"),
            make_doc("1", "A", "2026-01-01T00:00:00+00:00", "2026-12-31T23:59:59+00:00"),
        ]
        result = engine._filter(docs, "événements en 2026")
        assert result[0].metadata["title"] == "A"
        assert result[1].metadata["title"] == "C"


# ── Tests _build_context ──────────────────────────────────────────────────────

class TestBuildContext:
    def test_contains_title(self, engine):
        doc = make_doc("1", "Mon Concert", "2026-01-01T00:00:00+00:00", "2026-01-01T23:59:59+00:00")
        context = engine._build_context([doc])
        assert "Mon Concert" in context

    def test_contains_description(self, engine):
        doc = make_doc("1", "Concert", "2026-01-01T00:00:00+00:00", "2026-01-01T23:59:59+00:00",
                       page_content="Super concert de jazz")
        context = engine._build_context([doc])
        assert "Super concert de jazz" in context

    def test_contains_separator(self, engine):
        docs = [
            make_doc("1", "A", "2026-01-01T00:00:00+00:00", "2026-01-01T23:59:59+00:00"),
            make_doc("2", "B", "2026-02-01T00:00:00+00:00", "2026-02-01T23:59:59+00:00"),
        ]
        context = engine._build_context(docs)
        assert "---" in context

    def test_respects_max_events(self, engine):
        engine.config.max_events = 2
        docs = [make_doc(str(i), f"Event {i}", "2026-01-01T00:00:00+00:00", "2026-01-01T23:59:59+00:00") for i in range(5)]
        context = engine._build_context(docs)
        # Max 2 événements → "Event 0" et "Event 1" présents, "Event 4" absent
        assert "Event 0" in context
        assert "Event 4" not in context

    def test_empty_docs_returns_empty_string(self, engine):
        assert engine._build_context([]) == ""


# ── Tests ask ─────────────────────────────────────────────────────────────────

class TestAsk:
    def test_empty_question_returns_error(self, engine):
        result = engine.ask("")
        assert "answer" in result
        assert result["events"] == []

    def test_whitespace_question_returns_error(self, engine):
        result = engine.ask("   ")
        assert "answer" in result
        assert result["events"] == []

    def test_no_results_returns_empty_events(self, engine):
        engine.vs.similarity_search_with_relevance_scores.return_value = []
        result = engine.ask("quelque chose d'introuvable en mars 2026")
        assert "events" in result
        assert result["events"] == []

    def test_valid_question_calls_llm(self, engine):
        doc = make_doc("1", "Concert", "2026-01-17T19:00:00+00:00", "2026-01-17T21:00:00+00:00",
                       page_content="Jazz au Pan Piper")
        engine.vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.85)]
        engine.llm.invoke.return_value = MagicMock(content="Voici un concert jazz...")
        result = engine.ask("concerts jazz à Paris en janvier 2026")
        assert engine.llm.invoke.called
        assert result["answer"] == "Voici un concert jazz..."
        assert len(result["events"]) == 1

    def test_events_payload_structure(self, engine):
        doc = make_doc("uid42", "Spectacle", "2026-05-19T17:30:00+00:00", "2026-06-20T22:00:00+00:00")
        engine.vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.90)]
        engine.llm.invoke.return_value = MagicMock(content="Réponse test")
        result = engine.ask("théâtre en mai 2026")
        event = result["events"][0]
        for key in ("uid", "title", "start", "end", "location_name", "city", "lien"):
            assert key in event
