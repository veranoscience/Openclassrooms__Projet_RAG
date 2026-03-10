from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

input_file = Path("data/processed/events_paris_cleaned.csv")
vectorstore_file = Path("vectorstores/faiss_index")
preview_file = Path("vectorstores/preview.txt")

embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
chunk_size = 800
chunk_overlap = 100

def load_dataset(input_file: Path) -> pd.DataFrame:
    return pd.read_csv(input_file)

def row_to_documents(row: pd.Series) -> Document:
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
    return Document(page_content=content, metadata=metadata)

def build_documents(df: pd.DataFrame) -> list[Document]:
    return [row_to_documents(row) for _, row in df.iterrows()]


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])
    return text_splitter.split_documents(documents)

def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name, 
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)

def save_vectorstore(vectorstore: FAISS, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_file))

def save_preview(results: list[Document], output_file: Path, query: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"Query de test: {query}\n\n")
        for i, doc in enumerate(results, start=1):
            f.write(f"Resultat {i}:\n")
            f.write(f"Contenu: {doc.page_content}\n")
            f.write(f"Metadata: {doc.metadata}\n\n")

def main() -> None:
    print("Construction de l'index FAISS...")

    df= load_dataset(input_file)
    print(f"Dataset chargé avec {len(df)} d'événements")

    documents = build_documents(df)
    print(f"{len(documents)} documents construits à paartir du dataset")

    chunks = split_documents(documents)
    print(f"{len(chunks)} chunks crées à partir des documents")

    embeddings = build_embeddings()
    print(f"Modèle d'embeddings '{embedding_model_name}' chargé")

    vectorstore = build_vectorstore(chunks, embeddings)
    print("Index FAISS construit avec succès")

    save_vectorstore(vectorstore, vectorstore_file)
    print(f"Index FAISS sauvegardé dans '{vectorstore_file}'")

    test_query = "Quelles sont les événemets cuturels à Paris en mar 2026?"
    results = vectorstore.similarity_search(test_query, k=3)

    print(f"Résultats de la recherche pour la requête: '{test_query}'")
    print(f"Nombre de résultasts: {len(results)}")

    for i, doc in enumerate(results, start=1):
        print(f"Resultat: {i}")
        print(f"Titre: {doc.metadata.get("titre")}")
        print(f"Ville: {doc.metadata.get("city")}")
        print(f"Date début: {doc.metadata.get("firstdate_begin")}")
        print(f"URL: {doc.metadata.get("canonicaurl")}")
        print(f"Contenu: {doc.page_content[:200]} ...\n")

    save_preview(results, preview_file, test_query)
    print(f"Aperçu de résultasts sauvergardé dans '{preview_file}'")

if __name__ == "__main__":
    main()

