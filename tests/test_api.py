"""
Tests unitaires — API FastAPI
Couvre : GET /health, POST /ask, POST /rebuild
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Fixture : client avec engine mocké ────────────────────────────────────────

@pytest.fixture
def client():
    """Crée un TestClient en mockant RAGEngine (pas de FAISS ni Mistral réels)."""
    with (
        patch("langchain_community.vectorstores.FAISS.load_local"),
        patch("langchain_huggingface.HuggingFaceEmbeddings"),
        patch("langchain_mistralai.ChatMistralAI"),
    ):
        from api.main import app
        import api.main as main_module

        mock_engine = MagicMock()
        mock_engine.config = MagicMock()
        mock_engine.config.retrieve_k = 80
        mock_engine.config.max_events = 8
        mock_engine.ask.return_value = {
            "answer": "Voici les événements trouvés.",
            "events": [
                {
                    "uid": "abc123",
                    "title": "Concert Jazz",
                    "start": "2026-01-17T19:00:00+00:00",
                    "end": "2026-01-17T21:00:00+00:00",
                    "location_name": "Le Pan Piper",
                    "location_address": "2 Impasse Lamier",
                    "city": "Paris",
                    "lien": "https://openagenda.com/culture/events/pegazz",
                    "originagenda_title": "Culture",
                }
            ],
        }

        # Injecter l'engine mocké dans le module
        main_module._engine = mock_engine

        with TestClient(app) as c:
            yield c

        # Nettoyer l'état global entre les tests
        main_module._engine = None


# ── Tests /health ─────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


# ── Tests /ask ────────────────────────────────────────────────────────────────

class TestAsk:
    def test_ask_returns_200(self, client):
        response = client.post("/ask", json={"question": "concerts jazz à Paris en janvier 2026"})
        assert response.status_code == 200

    def test_ask_response_has_answer(self, client):
        response = client.post("/ask", json={"question": "concerts jazz à Paris en janvier 2026"})
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)

    def test_ask_response_has_events(self, client):
        response = client.post("/ask", json={"question": "concerts jazz à Paris en janvier 2026"})
        data = response.json()
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_ask_event_structure(self, client):
        response = client.post("/ask", json={"question": "concerts jazz à Paris en janvier 2026"})
        event = response.json()["events"][0]
        for field in ("uid", "title", "start", "end", "location_name", "city", "lien"):
            assert field in event

    def test_ask_with_custom_top_k(self, client):
        response = client.post("/ask", json={
            "question": "théâtre en mai 2026",
            "top_k": 50,
            "max_events": 5,
        })
        assert response.status_code == 200

    def test_ask_empty_question_returns_422(self, client):
        """Pydantic valide min_length=1 : une question vide doit être rejetée."""
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_ask_missing_question_returns_422(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_ask_top_k_out_of_range_returns_422(self, client):
        """top_k doit être entre 1 et 200."""
        response = client.post("/ask", json={"question": "jazz", "top_k": 300})
        assert response.status_code == 422

    def test_ask_max_events_out_of_range_returns_422(self, client):
        """max_events doit être entre 1 et 20."""
        response = client.post("/ask", json={"question": "jazz", "max_events": 50})
        assert response.status_code == 422


# ── Tests /rebuild ────────────────────────────────────────────────────────────

class TestRebuild:
    def test_rebuild_index_only_returns_ok(self, client):
        with patch("api.main.rebuild") as mock_rebuild:
            mock_rebuild.return_value = None
            response = client.post("/rebuild", json={"mode": "index_only", "force": True})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["mode"] == "index_only"

    def test_rebuild_full_returns_ok(self, client):
        with patch("api.main.rebuild") as mock_rebuild:
            mock_rebuild.return_value = None
            response = client.post("/rebuild", json={"mode": "full", "force": False})
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_rebuild_called_with_correct_args(self, client):
        with patch("api.main.rebuild") as mock_rebuild:
            mock_rebuild.return_value = None
            client.post("/rebuild", json={"mode": "index_only", "force": True})
            mock_rebuild.assert_called_once_with(mode="index_only", force=True)

    def test_rebuild_error_returns_500(self, client):
        with patch("api.main.rebuild") as mock_rebuild:
            mock_rebuild.side_effect = RuntimeError("Erreur de reconstruction")
            response = client.post("/rebuild", json={"mode": "index_only", "force": True})
            assert response.status_code == 500
            assert response.json()["status"] == "error"

    def test_rebuild_invalid_token_returns_401(self, client):
        with patch.dict(os.environ, {"REBUILD_TOKEN": "secret123"}):
            response = client.post(
                "/rebuild",
                json={"mode": "index_only", "force": True},
                headers={"x-rebuild-token": "mauvais-token"},
            )
            assert response.status_code == 401

    def test_rebuild_valid_token_accepted(self, client):
        with (
            patch.dict(os.environ, {"REBUILD_TOKEN": "secret123"}),
            patch("api.main.rebuild") as mock_rebuild,
        ):
            mock_rebuild.return_value = None
            response = client.post(
                "/rebuild",
                json={"mode": "index_only", "force": True},
                headers={"x-rebuild-token": "secret123"},
            )
            assert response.status_code == 200

    def test_rebuild_no_token_required(self, client):
        """Sans REBUILD_TOKEN configuré, toute requête est acceptée."""
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("api.main.rebuild") as mock_rebuild,
        ):
            os.environ.pop("REBUILD_TOKEN", None)
            mock_rebuild.return_value = None
            response = client.post("/rebuild", json={"mode": "index_only", "force": True})
            assert response.status_code == 200
