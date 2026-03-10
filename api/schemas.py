from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

class AskRequest (BaseModel):
    question: str = Field(..., min_length=1, description="Question posée pas l'utilisateur")
    top_k: int = Field (80, ge=1, le=200, description="Nombre de documents récupérés avant filtrage")
    max_events: int = Field(8, ge=1, le=20, description="Nombre d'événements max renvoyés")

class EventItem(BaseModel):
    uid: Optional[str] = None
    title: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    location_name: Optional[str] = None
    location_address: Optional[str] = None
    city: Optional[str] = None
    lien: Optional[str] = None
    originagenda_title: Optional[str] = None

# Réponse du RAG
class AskResponse(BaseModel):
    answer: str
    events: List[EventItem]

class RebuildRequest(BaseModel):
    mode: str = Field("index_only", description="index_only | full")
    force: bool = Field(True, description="Reconstruire même si l’index existe")


