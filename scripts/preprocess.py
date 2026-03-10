from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd

input_file = Path("data/raw/events_paris.json")
output_dir = Path("data/processed")
output_file = output_dir / "events_paris_cleaned.csv"

columns_to_keep = [
    "uid",
    "slug",
    "canonicalurl",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "conditions_fr",
    "keywords_fr",
    "updatedat",
    "daterange_fr",
    "firstdate_begin",
    "firstdate_end",
    "lastdate_begin",
    "lastdate_end",
    "location_uid",
    "location_name",
    "location_address",
    "location_district",
    "location_insee",
    "location_postalcode",
    "location_city",
    "location_department",
    "location_region",
    "location_countrycode",
    "originagenda_uid",
    "originagenda_title",
    "age_min",
    "age_max",
    "status",
    "timings",
]

def load_data(input_file: Path) -> list[dict[str, Any]]:
    """ Charge les données à partir du fichier JSON et retourne une list de dictionnaires"""
    with input_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("results", [])

def clean_text(text:Any) -> str:
    """ Nettoie le texte en supprimant les balises HTML, les espaces"""
    if text is None:
        return ""
    
    text = str(text)
    text = unescape(text) # convertir les entités HTML en caractères normaux
    text = re.sub(r"<[^>]+>", " ", text) # supprimer les balises HTML
    text = re.sub(r"\s+", " ", text).strip() # suprimes les espaces en trop
    return text

def normalize_data(value: Any) -> str:
    """ Transforme une liste en une chaine de caractères lisibles"""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(x).strip() for x in value if str(x).strip())
    return str(value)

def normalize_text(value: Any) -> str:
    """ Nettoie et normalise le texte"""
    if value is None:
        return""
    return str(value).strip()

def build_event_text(row: pd.Series) -> str:
    """ Construit un texte descriptif enrichi pour la vectorisation/idexation"""
    parts = [
         f"Titre : {row['title_fr']}",
        f"Description courte : {row['description_fr']}",
        f"Description longue : {row['longdescription_fr']}",
        f"Mots-clés : {row['keywords_fr']}",
        f"Conditions : {row['conditions_fr']}",
        f"Lieu : {row['location_name']}",
        f"Adresse : {row['location_address']}",
        f"Ville : {row['location_city']}",
        f"Département : {row['location_department']}",
        f"Région : {row['location_region']}",
        f"Agenda source : {row['originagenda_title']}",
        f"Date de début : {row['firstdate_begin']}",
        f"Date de fin : {row['lastdate_end']}",
        f"URL : {row['canonicalurl']}",
    ]

    cleaned_parts = [part for part in parts if part.split(":",1)[1].strip()]
    return "\n".join(cleaned_parts)

def preprocess_data(events: list[dict[str, Any]]) -> pd.DataFrame:
    """ Convertit la liste de dictionnaires en DataFrame, nettoie et normalise les données"""
    df = pd.DataFrame(events)
    
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        print(f"Colonnes absentes dans les données : {missing_columns}")
    
    available_cols = [col for col in columns_to_keep if col in df.columns]
    df = df[available_cols].copy()

    text_columns =  [
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "conditions_fr",
        "location_name",
        "location_address",
        "location_district",
        "location_city",
        "location_department",
        "location_region",
        "originagenda_title",
        "canonicalurl",
        "slug",
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
    
    if "description_fr" in df.columns:
        df["description_fr"] = df["description_fr"].apply(clean_text)
    if "longdescription_fr" in df.columns:
        df["longdescription_fr"] = df["longdescription_fr"].apply(clean_text)
    if "keywords_fr" in df.columns:
        df["keywords_fr"] = df["keywords_fr"].apply(normalize_data)
    data_columns = [
        "updatedat",
        "firstdate_begin",
        "firstdate_end",
        "lastdate_begin",
        "lastdate_end",
    ]

    for col in data_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    
    avant_dedup = len(df)
    if "uid" in df.columns:
        df = df.drop_duplicates(subset=["uid"]).copy()
    apres_dedup = len(df)
    

    df["event_text"] = df.apply(build_event_text, axis=1)

    if "title_fr" in df.columns:
        df["title_fr"] = df["title_fr"].fillna("")
    if "location_city" in df.columns:
        df["location_city"] = df["location_city"].fillna("")
    
    print(f"Prétraiement terminé, nombre d'événements final: {len(df)}")
    print (f"Nombre d'événements avant déduplication: {avant_dedup}, après déduplication; {apres_dedup}")
    print (f"Nombre de colonnes finales: {len(df.columns)}")
    return df

def save_data(df: pd.DataFrame, output_file: Path) -> None:
    """ Enregistre le DataFrame nettoyé au format CSV"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Données prétraitées enregistrés dans: {output_file}")

def main() -> None:
    events = load_data(input_file)
    print (f"Nombre d'évenements chergés: {len(events)}")

    df = preprocess_data(events)
    save_data(df, output_file)

    print()
    print("Aperçu des données prétraitées:")
    print(df.head(3))
    print("Apperçu les colonnes:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()
        

