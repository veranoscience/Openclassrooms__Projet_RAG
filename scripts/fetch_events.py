from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv(
    "ODS_BASE_URL",
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets",

)

dataset = os.getenv("ODS_DATASET", "evenements-publics-openagenda")
location = os.getenv("OPENAGENDA_LOCATION", "Paris")
past_days = int(os.getenv("OPENAGENDA_PAST_DAYS", "365"))
future_days = int(os.getenv("OPENAGENDA_FUTURE_DAYS", "90"))
page_size = int(os.getenv("OPENAGENDA_PAGE_SIZE", "100"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data/raw"))
OUTPUT_FILE = OUTPUT_DIR / "events_paris.json"

def compute_date_range() -> tuple[str, str]:
    """ Calcule la plage de dates:
    - date de début: aujourd'hui - past_days
    - date de fin: aujourd'hui + future_days
    Retourne les date au format YYYY-MM-DD
    """
    today = date.today()
    start_date = today - timedelta(days=past_days)
    end_date = today + timedelta(days=future_days)
    return start_date.isoformat(), end_date.isoformat()

def buid_where_clause (location: str, start_date:str, end_date: str) -> str:
    """ Construit la clause where pour le filre de raquète API
    - les evenements localisé à Paris
    - dont la périonde recouvre notre fenêtre temporelle
    - ayant un agenda qui n'est pas vide
    """
    where_clause = (
        f'location_city = "{location}" '
        f"and originagenda_uid is not null "
        f"and lastdate_end >= date'{start_date}' "
        f"and firstdate_begin <= date'{end_date}'"
    )
    return where_clause

def fetch_page (offset: int, limit: int, where_clause: str) -> dict [str, Any]:
    """ Récupère une page de résultats de l'API
    - offset: nombre d'éléments à sauter
    - limit: nombre d'éléments à récupérer
    - where_clause: clause where pour filtrer les résultats
    Retourne la réponse de l'API sous forme de dictionnaire
    """
    url = f"{base_url}/{dataset}/records"
    params = {
        "where": where_clause,
        "limit": limit,
        "offset": offset,
        "order_by": "firstdate_begin desc",
        "language": "fr",
        "timezone": "Europe/Paris",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def fetch_all_events() -> dict[str, Any]:
    """ Récupère toutes les pages de résultats et renvoie:
    - le nombre total d'événements
    - les évenements sous forme de liste
    - les paramètres de la requete utilisée
    """
    start_date, end_date = compute_date_range()
    where_clause = buid_where_clause(location, start_date, end_date)

    print (f"Fetching events en ville de {location} du {start_date} au {end_date}")
    print (f"Where clause: {where_clause}")
    print (f"Page size: {page_size}")
    print()

    all_results: list [dict[str, Any]] = []
    offset = 0
    total_count = None

    while True:
        print (f"Fetching page avec offset {offset}")
        payload = fetch_page(offset=offset, limit=page_size, where_clause=where_clause)

        if total_count is None:
            total_count = payload.get("total_count", 0)
            print (f"Total events matching criteria: {total_count}")

        results = payload.get("results", [])
        if not results:
            print ("Il n'y a plus de résultats à récupéerer")
            break

        all_results.extend(results)
        print (f"Page récupérée: offset: {offset}, nombre d'événements dans la page: {len(results)}")
        offset += page_size

        if len(all_results) >= total_count:
            print ("Tous les événements ont été récupérés")
            break

    print()
    print(f"Total récupére: {len(all_results)} événements")

    return {
        "dataset": dataset,
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "page_size": page_size,
        "where_clause": where_clause,
        "total_count": total_count,
        "results": all_results,
    }

def save_results(payload: dict[str, Any], OUTPUT_FILE: Path) -> None:
    """ Sauvegarde les résultast dans un fichier JSON"""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

        print (f"Résultats sauvergarsés dans {OUTPUT_FILE}")


def main():
    try:
        payload = fetch_all_events()
        save_results(payload, OUTPUT_FILE)
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des données: {e}")
        raise
    except requests.HTTPError as e:
        print (f"Erreur HTTP: {e}")
        raise
    except Exception as e:
        print (f"Erreur innatendue: {e}")
        raise

if __name__ == "__main__":
    main()
        




