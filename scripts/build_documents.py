from __future__ import annotations

from pathlib import Path
import pandas as pd
from langchain_core.documents import Document

input_file = Path("data/processed/events_paris_cleaned.csv")
output_preview_file = Path("data/processed/documents_preview.txt")

def load_dataset(input_file: Path) -> pd.DataFrame:
    """ Chargé le dataset"""
    return pd.read_csv(input_file)

def create_documents(row: pd.Series) -> Document:
    """ Transforme une ligne du dataset en Document LangChain"""

    metadata = {
    
        "uid": row.get("uid", ""),
        "title": row.get("title_fr", ""),
        "city": row.get("location_city", ""),
        "department": row.get("location_department", ""),
        "region": row.get("location_region", ""),
        "location_name": row.get("location_name", ""),
        "location_address": row.get("location_address", ""),
        "originagenda_title": row.get("originagenda_title", ""),
        "canonicalurl": row.get("canonicalurl", ""),
        "firstdate_begin": str(row.get("firstdate_begin", "")),
        "lastdate_end": str(row.get("lastdate_end", "")),

    }
    content = row.get("event_text", "")

    return Document (page_content=content, metadata=metadata)

def build_documents(df: pd.DataFrame) -> list[Document]:
    """ Transforme le DataFrame en une liste de Documents"""
    documents = []
    for _, row in df.iterrows():
        doc = create_documents(row)
        documents.append(doc)
    return documents

def save_preview(documents: list[Document], output_file: Path, n: int = 3) -> None:
    """ Sauvegarde un aperçu des decuments dans un fichier texte"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for i, doc in enumerate (documents[:n], start=1):
            f.write(f"Document{i}:\n")
            f.write(f"Metadata: {doc.metadata}\n")
            f.write(f"Content:\n{doc.page_content}\n")

def main():
    df = load_dataset(input_file)
    print(f"Dataset chargé avec {len(df)} lignes")

    documents = build_documents(df)
    print(f"{len(documents)} documents créés")

    if documents:
        print("\nAperçu du premier document:")
        print("Metadata:", documents[0].metadata)
        print("Contet (début):", documents[0].page_content[:500], "...\n")

    save_preview(documents, output_preview_file)
    print(f"Aperçu sauvergardé dans: {output_preview_file}")

if __name__ == "__main__":
    main()
    