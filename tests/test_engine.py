"""
Tests unitaires — RAGEngine
Couvre : parse_date, _retrieve, _filter, ask
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document

from rag.engine import parse_date, RAGConfig, RAGEngine


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


def test_month_year():
    """Récupération — parse_date extrait correctement un mois et une année"""
    result = parse_date("concerts jazz à Paris en janvier 2026")
    assert result is not None
    start, end = result
    assert start == pd.Timestamp(2026, 1, 1, tz="UTC")
    assert end == pd.Timestamp(2026, 1, 31, 23, 59, 59, tz="UTC")


def test_filters_by_score_threshold(engine):
    """Récupération — seuls les documents au-dessus du seuil 0.3 sont retenus"""
    doc_high = make_doc("1", "Pertinent", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
    doc_low  = make_doc("2", "Hors sujet", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
    engine.vs.similarity_search_with_relevance_scores.return_value = [
        (doc_high, 0.85),
        (doc_low,  0.10),
    ]
    engine.config.score_threshold = 0.3
    result = engine._retrieve("concert jazz Paris")
    assert len(result) == 1
    assert result[0].metadata["title"] == "Pertinent"


def test_temporal_filter_keeps_matching(engine):
    """Récupération — le filtre temporel ne garde que les événements qui chevauchent la période"""
    doc_march = make_doc("1", "Mars",    "2026-03-01T00:00:00+00:00", "2026-03-31T23:59:59+00:00")
    doc_jan   = make_doc("2", "Janvier", "2026-01-01T00:00:00+00:00", "2026-01-31T23:59:59+00:00")
    result = engine._filter([doc_march, doc_jan], "événements en mars 2026")
    titles = [d.metadata["title"] for d in result]
    assert "Mars" in titles
    assert "Janvier" not in titles


def test_valid_question_calls_llm(engine):
    """Interrogation + Génération — une question valide déclenche l'appel au LLM"""
    doc = make_doc("1", "Concert", "2026-01-17T19:00:00+00:00", "2026-01-17T21:00:00+00:00",
                   page_content="Jazz au Pan Piper")
    engine.vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.85)]
    engine.llm.invoke.return_value = MagicMock(content="Voici un concert jazz...")
    result = engine.ask("concerts jazz à Paris en janvier 2026")
    assert engine.llm.invoke.called
    assert result["answer"] == "Voici un concert jazz..."
    assert len(result["events"]) == 1


def test_no_results_returns_empty_events(engine):
    """Génération — sans documents pertinents, la réponse ne contient pas d'événements inventés"""
    engine.vs.similarity_search_with_relevance_scores.return_value = []
    result = engine.ask("quelque chose d'introuvable en mars 2026")
    assert "events" in result
    assert result["events"] == []
