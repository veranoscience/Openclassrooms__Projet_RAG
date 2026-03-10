"""
Tests unitaires — Nettoyage / Prétraitement des données
Couvre : clean_text, normalize_data, normalize_text,
         build_event_text, preprocess_data
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from scripts.preprocess import (
    clean_text,
    normalize_data,
    normalize_text,
    build_event_text,
    preprocess_data,
)


# ── Fixture : événement minimal ───────────────────────────────────────────────

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


# ── Tests clean_text ──────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_html_tags(self):
        result = clean_text("<p>Bonjour <b>monde</b></p>")
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Bonjour" in result
        assert "monde" in result

    def test_decodes_html_entities(self):
        result = clean_text("Prix &lt;20&euro; &amp; entrée libre")
        assert "&lt;" not in result
        assert "&amp;" not in result

    def test_collapses_whitespace(self):
        result = clean_text("  texte   avec   espaces  ")
        assert "  " not in result
        assert result == result.strip()

    def test_none_returns_empty_string(self):
        assert clean_text(None) == ""

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_plain_text_unchanged(self):
        result = clean_text("Concert de jazz")
        assert result == "Concert de jazz"


# ── Tests normalize_data ──────────────────────────────────────────────────────

class TestNormalizeData:
    def test_list_to_comma_string(self):
        result = normalize_data(["jazz", "musique", "Paris"])
        assert result == "jazz, musique, Paris"

    def test_none_returns_empty(self):
        assert normalize_data(None) == ""

    def test_string_passthrough(self):
        assert normalize_data("jazz") == "jazz"

    def test_empty_list_returns_empty(self):
        assert normalize_data([]) == ""

    def test_list_with_spaces_stripped(self):
        result = normalize_data(["  jazz  ", " musique "])
        assert "jazz" in result
        assert "musique" in result
        assert "  " not in result


# ── Tests normalize_text ──────────────────────────────────────────────────────

class TestNormalizeText:
    def test_strips_whitespace(self):
        assert normalize_text("  Concert  ") == "Concert"

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_string_passthrough(self):
        assert normalize_text("Concert Jazz") == "Concert Jazz"


# ── Tests build_event_text ────────────────────────────────────────────────────

class TestBuildEventText:
    def test_contains_title(self, sample_event):
        row = pd.Series(sample_event)
        # Rename pour correspondre aux colonnes attendues par build_event_text
        row["title_fr"] = "Concert Jazz"
        row["description_fr"] = "Super concert"
        row["longdescription_fr"] = ""
        row["keywords_fr"] = "jazz, musique"
        row["conditions_fr"] = "Entrée libre"
        row["location_name"] = "Le Pan Piper"
        row["location_address"] = "2 Impasse Lamier"
        row["location_city"] = "Paris"
        row["location_department"] = "Paris"
        row["location_region"] = "Île-de-France"
        row["originagenda_title"] = "Culture"
        row["firstdate_begin"] = "2026-01-17"
        row["lastdate_end"] = "2026-01-17"
        row["canonicalurl"] = "https://example.com"
        text = build_event_text(row)
        assert "Concert Jazz" in text

    def test_contains_location(self, sample_event):
        row = pd.Series(sample_event)
        row["title_fr"] = "Expo"
        row["description_fr"] = ""
        row["longdescription_fr"] = ""
        row["keywords_fr"] = ""
        row["conditions_fr"] = ""
        row["location_name"] = "Musée test"
        row["location_address"] = "1 rue test"
        row["location_city"] = "Paris"
        row["location_department"] = "Paris"
        row["location_region"] = "IDF"
        row["originagenda_title"] = "Culture"
        row["firstdate_begin"] = "2026-01-01"
        row["lastdate_end"] = "2026-01-31"
        row["canonicalurl"] = "https://example.com"
        text = build_event_text(row)
        assert "Musée test" in text

    def test_empty_fields_excluded(self, sample_event):
        """Les champs vides ne doivent pas générer de ligne 'Champ : '."""
        row = pd.Series(sample_event)
        row["title_fr"] = "Expo"
        row["description_fr"] = ""
        row["longdescription_fr"] = ""
        row["keywords_fr"] = ""
        row["conditions_fr"] = ""
        row["location_name"] = "Lieu"
        row["location_address"] = ""
        row["location_city"] = "Paris"
        row["location_department"] = ""
        row["location_region"] = ""
        row["originagenda_title"] = ""
        row["firstdate_begin"] = "2026-01-01"
        row["lastdate_end"] = "2026-01-31"
        row["canonicalurl"] = "https://example.com"
        text = build_event_text(row)
        # Aucune ligne ne doit se terminer par ": " (champ vide inclus)
        for line in text.splitlines():
            assert not line.endswith(": ")


# ── Tests preprocess_data ─────────────────────────────────────────────────────

class TestPreprocessData:
    def test_returns_dataframe(self, sample_event):
        df = preprocess_data([sample_event])
        assert isinstance(df, pd.DataFrame)

    def test_event_text_column_created(self, sample_event):
        df = preprocess_data([sample_event])
        assert "event_text" in df.columns

    def test_event_text_not_empty(self, sample_event):
        df = preprocess_data([sample_event])
        assert df["event_text"].iloc[0].strip() != ""

    def test_html_cleaned_in_description(self, sample_event):
        df = preprocess_data([sample_event])
        desc = df["description_fr"].iloc[0]
        assert "<p>" not in desc
        assert "<b>" not in desc

    def test_dates_parsed_to_datetime(self, sample_event):
        df = preprocess_data([sample_event])
        assert pd.api.types.is_datetime64_any_dtype(df["firstdate_begin"])

    def test_deduplication_removes_duplicates(self, sample_event):
        df = preprocess_data([sample_event, sample_event])
        assert len(df) == 1

    def test_keywords_normalized_from_list(self, sample_event):
        df = preprocess_data([sample_event])
        keywords = df["keywords_fr"].iloc[0]
        assert isinstance(keywords, str)
        assert "jazz" in keywords

    def test_single_event_processed(self, sample_event):
        """Un seul événement valide doit produire un DataFrame avec 1 ligne."""
        df = preprocess_data([sample_event])
        assert len(df) == 1
