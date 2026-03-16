from __future__ import annotations

import calendar
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

load_dotenv()

# Parsing de la date d'événement
months = {
    "janvier": 1,
    "février": 2, "fevrier": 2, 
    "mars": 3,
    "avril": 4,
    "mai": 5, 
    "juin": 6,
    "juillet": 7,
    "août": 8, "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12, "decembre": 12,
}

def parse_date(question: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    # Recherche de la date dans la question
    q = question.lower()

    #  Date complète : "12 mars 2026" / "1er avril 2026"
    month_keys = "|".join(map(re.escape, months.keys()))
    m_day = re.search(rf"\b(\d{{1,2}})(?:er)?\s+({month_keys})\s+(20\d{{2}})\b", q)
    if m_day:
        day = int(m_day.group(1))
        month = months[m_day.group(2)]
        year = int(m_day.group(3))
        try:
            start = pd.Timestamp(year, month, day, tz="UTC")
        except ValueError:
            return None  # ex: 31 février
        end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return start, end

    # Mois + année : "mars 2026"
    m_month = re.search(rf"\b({month_keys})\b\s*(20\d{{2}})\b", q)
    if m_month:
        month = months[m_month.group(1)]
        year = int(m_month.group(2))
        last_day = calendar.monthrange(year, month)[1]
        start = pd.Timestamp(year, month, 1, tz="UTC")
        end = pd.Timestamp(year, month, last_day, 23, 59, 59, tz="UTC")
        return start, end

    # Année seule : "en 2026" / "2026"
    m_year = re.search(r"\b(20\d{2})\b", q)
    if m_year:
        year = int(m_year.group(1))
        start = pd.Timestamp(year, 1, 1, tz="UTC")
        end = pd.Timestamp(year, 12, 31, 23, 59, 59, tz="UTC")
        return start, end

    return None

def overlaps(doc: Document, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    doc_start = pd.to_datetime(doc.metadata.get("firstdate_begin"), utc=True, errors="coerce")
    doc_end = pd.to_datetime(doc.metadata.get("lastdate_end"), utc=True, errors="coerce")
    if pd.isna(doc_start) or pd.isna(doc_end):
        return False
    return not (doc_end < start or doc_start > end)

def sort_by_start_date(docs: list[Document]) -> list[Document]:
    def get_start_date(doc: Document):
        return pd.to_datetime(doc.metadata.get("firstdate_begin"), utc=True, errors="coerce")
    return sorted(docs, key=get_start_date)

def deduplicate_docs(docs: list[Document]) -> list[Document]:
    seen = set()
    out = []
    for doc in docs:
        uid = doc.metadata.get("uid")
        if uid in seen:
            continue
        seen.add(uid)
        out.append(doc)
    return out

system_prompt = """Tu es un assistant de recommandation culturelle pour Paris
Nous sommes le {current_date}

RÈGLE GÉOGRAPHIQUE :
Si l'utilisateur mentionne une ville hors Paris (Lyon, Marseille, Bordeaux...), REFUSE DE RÉPONDRE.
Dis : "Désolé, je ne couvre que Paris"

RÈGLES STRICTES :
1. Utilise UNIQUEMENT les informations présentes dans le contexte.
2. N'invente JAMAIS d'informations absentes du contexte.
3. Si le contexte ne contient pas d'événements correspondant à la question, réponds : "Je n'ai pas trouvé d'événements correspondant à votre demande."
4. Réponds en français avec 1 à 2 phrases d'introduction seulement — ne liste pas les événements.
"""
prompt_template = """Question utilisateur: {question}

Contexte (événements retrouvés dans la base) :
{context}

Réponse (1 à 2 phrases d'introduction uniquement — ne liste pas les événements, ils seront affichés automatiquement par l'interface) :"""

@dataclass
class RAGConfig:
    faiss_dir: str = "vectorstores/faiss_index"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    mistral_model: str = "mistral-large-latest"
    temperature: float = 0.1
    retrieve_k: int = 80
    max_events: int = 8
    score_threshold: float = 0.3   # seuil de similarité cosinus (0-1) pour réduire le bruit

class RAGEngine:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()

        self.embedding = HuggingFaceEmbeddings(
            model_name = self.config.embedding_model,
            encode_kwargs = {"normalize_embeddings": True},
        )
        # load_local utile pickle -> flaq requis pour éviter de réindexer à chaque lancement
        self.vs = FAISS.load_local(self.config.faiss_dir, self.embedding, allow_dangerous_deserialization=True)

        self.llm = ChatMistralAI(
            model = os.getenv("MISTRAL_MODEL", self.config.mistral_model), 
            temperature=self.config.temperature,
            max_retries =2,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", prompt_template),
            ]
        )
    def _retrieve(self, question: str) -> list[Document]:
        # Retrieval avec seuil de score pour réduire le bruit
        docs_and_scores = self.vs.similarity_search_with_relevance_scores(
            question, k=self.config.retrieve_k
        )
        return [doc for doc, score in docs_and_scores if score >= self.config.score_threshold]
    
    def _filter(self, docs: list[Document], question: str) -> list[Document]:
        # Filtrage temporel
        parsed = parse_date(question)
        if parsed:
            start, end = parsed
            docs = [doc for doc in docs if overlaps(doc, start, end)]

        # Tri de déduplication
        docs = deduplicate_docs(docs)

        # Tri par date de début
        docs = sort_by_start_date(docs)
        return docs
        
    def _build_context(self, docs: list[Document]) -> str:
        # Construction du contexte à partir des métadonnées + description (page_content)
        lines: list[str] = []
        for doc in docs[:self.config.max_events]:
            md = doc.metadata
            description = doc.page_content[:300].strip() if doc.page_content else ""
            lines.append(
                "\n".join(
                    [
                        f"Titre: {md.get('title', '')}",
                        f"Début: {md.get('firstdate_begin','')}",
                        f"Fin: {md.get('lastdate_end','')}",
                        f"Lieu: {md.get('location_name','')}",
                        f"Adresse: {md.get('location_address','')}",
                        f"Ville: {md.get('city','')}",
                        f"Lien: {md.get('canonicalurl','')}",
                        f"Description: {description}",
                    ]
                )
            )
            lines.append("---")
        return "\n".join(lines).strip()
    
    def ask(self, question: str) -> str:
        if not question or not question.strip():
            return {"answer": "Je n'ai pas compris la question. Veuillez reformuler.", "events": []}
        
        retrieved = self._retrieve(question)
        filtered = self._filter(retrieved, question)

        if not filtered:
            return {
                "answer": "Je n'ai pas trouvé d'évenemets correspondant à votre périod. Essaie d'élargir dériod ou une autre thématique",
                "events": [],
            }
        context = self._build_context(filtered)

        messages = self.prompt.format_messages(
            question=question,
            context=context,
            max_events=self.config.max_events,
            current_date=date.today().strftime("%d %B %Y"),
        )
        ai_msg = self.llm.invoke(messages)

        # Payload pour l'API (list d'événements)
        events = []
        for doc in filtered[: self.config.max_events]:
            md = doc.metadata
            events.append(
                {
                    "uid": str(md.get("uid")) if md.get("uid") is not None else None,
                    "title": md.get("title"),
                    "start": md.get("firstdate_begin"),
                    "end": md.get("lastdate_end"),
                    "location_name": md.get("location_name"),
                    "location_address": md.get("location_address"),
                    "city": md.get("city"),
                    "lien": md.get("canonicalurl"),
                    "originagenda_title": md.get("originagenda_title"),
                }
            )
            
        return {"answer": ai_msg.content, "events": events}

