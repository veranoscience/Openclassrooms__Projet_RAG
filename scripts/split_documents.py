from __future__ import annotations

from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

input_file = Path("data/processed/events_paris_cleaned.csv")
output_preview_file = Path("data/processed/chunks_preview.txt")

def load_dataset(input_file: Path) -> pd.DataFrame:
    return pd.read_csv(input_file)

def row_to_document(row: pd.Series) -> Document:
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
    conent = row.get("event_text", "")
    return Document(page_content=conent, metadata=metadata)

def build_documents(df: pd.DataFrame) -> list[Document]:
    return [row_to_document(row) for _, row in df.iterrows()]

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
    return text_splitter.split_documents(documents)

def save_preview(chunks: list[Document], output_file: Path, n: int=5) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks[:n], start=1):
            f.write(f"Chunk {i}:\n")
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write(f"Content:\n{chunk.page_content}\n\n")

def main() -> None:
    df = load_dataset(input_file)
    print(f"Dataset chargé avec {len(df)} lignes")

    documents = build_documents(df)
    print(f"Documents crées: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Chunks crées: {len(chunks)}")

    if documents:
        average_chunk = len(chunks) / len (documents)
        print(f"Nombre moyen de chunks par document: {average_chunk:.2f}")

    if chunks:
        print("\nAperçu du premier chunk:")
        print(f"Metadata: {chunks[0].metadata}")
        print (f"Content (début): {chunks[0].page_content[:500]}...\n")
    
    save_preview(chunks, output_preview_file)
    print(f"Aperçu des chunks sauvergardé dans : {output_preview_file}")

if __name__ == "__main__":
    main()