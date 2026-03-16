"""
Tests unitaires — Vectorisation / Indexation FAISS
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch


def test_embedding_model_name():
    """Vectorisation — le bon modèle HuggingFace multilingue doit être chargé"""
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
