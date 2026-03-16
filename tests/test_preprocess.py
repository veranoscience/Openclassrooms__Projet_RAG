"""
Tests unitaires — Nettoyage / Prétraitement des données
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from scripts.preprocess import clean_text, preprocess_data


@pytest.fixture
def sample_event():
    return {
        "uid": "evt001",
        "slug": "concert-jazz-paris",
        "canonicalurl": "https://openagenda.com/culture/events/concert-jazz",
        "title_fr": "Concert Jazz",
        "description_fr": "<p>Un <b>super</b> concert de jazz.</p>",
        "longdescription_fr": "Description longue de l'événement.",
        "conditions_fr": "Entrée libre",
        "keywords_fr": ["jazz", "musique", "Paris"],
        "updatedat": "2026-01-01T00:00:00+00:00",
        "daterange_fr": "17 janvier 2026",
        "firstdate_begin": "2026-01-17T19:00:00+00:00",
        "firstdate_end": "2026-01-17T21:00:00+00:00",
        "lastdate_begin": "2026-01-17T19:00:00+00:00",
        "lastdate_end": "2026-01-17T21:00:00+00:00",
        "location_uid": "loc001",
        "location_name": "Le Pan Piper",
        "location_address": "2 Impasse Lamier",
        "location_district": "11e",
        "location_insee": "75111",
        "location_postalcode": "75011",
        "location_city": "Paris",
        "location_department": "Paris",
        "location_region": "Île-de-France",
        "location_countrycode": "FR",
        "originagenda_uid": "agenda001",
        "originagenda_title": "Culture Paris",
        "age_min": None,
        "age_max": None,
        "status": 1,
        "timings": [],
    }


def test_removes_html_tags():
    """Nettoyage — les balises HTML doivent être supprimées"""
    result = clean_text("<p>Un <b>super</b> concert de jazz.</p>")
    assert "<p>" not in result
    assert "<b>" not in result
    assert "concert de jazz" in result


def test_html_cleaned_in_description(sample_event):
    """Nettoyage bout-en-bout — preprocess_data nettoie les descriptions HTML"""
    df = preprocess_data([sample_event])
    desc = df["description_fr"].iloc[0]
    assert "<p>" not in desc
    assert "<b>" not in desc
    assert "concert de jazz" in desc
