"""
Tests unitaires — API FastAPI
Couvre : POST /ask
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with (
        patch("langchain_community.vectorstores.FAISS.load_local"),
        patch("langchain_huggingface.HuggingFaceEmbeddings"),
        patch("langchain_mistralai.ChatMistralAI"),
    ):
        from api.main import app
        import api.main as main_module

        mock_engine = MagicMock()
        mock_engine.ask.return_value = {
            "answer": "Voici les événements trouvés.",
            "events": [{"uid": "abc", "title": "Concert Jazz", "start": "2026-01-17T19:00:00+00:00",
                        "end": "2026-01-17T21:00:00+00:00", "location_name": "Le Pan Piper",
                        "location_address": "2 Impasse Lamier", "city": "Paris",
                        "lien": "https://openagenda.com/events/abc", "originagenda_title": "Culture"}],
        }
        main_module._engine = mock_engine

        with TestClient(app) as c:
            yield c

        main_module._engine = None


def test_ask_returns_200(client):
    """API — /ask retourne 200 pour une question valide"""
    response = client.post("/ask", json={"question": "concerts jazz à Paris en janvier 2026"})
    assert response.status_code == 200


def test_ask_empty_question_returns_422(client):
    """API — Pydantic rejette une question vide avec 422"""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
