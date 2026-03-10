"""
Evaluation RAGAS : utilise Mistral pour juger la qualité du RAG
Métriques :
    faithfulness       – réponse fidèle au contexte récupéré ?
    answer_relevancy   – réponse pertinente par rapport à la question ?
    context_precision  – contexte récupéré précis (peu de bruit) ?
    context_recall     – infos clés bien récupérées ? (nécessite ground_truth)

"""
from __future__ import annotations

import argparse
import csv
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_mistralai import ChatMistralAI
from ragas import evaluate, SingleTurnSample
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall  # noqa: E402
# Note: ResponseRelevancy/AnswerRelevancy incompatible avec LangchainLLMWrapper + Mistral dans cette version de RAGAS

from rag.engine import RAGEngine, RAGConfig


def build_context(doc) -> str:
    """Convertit un document FAISS en texte pour le contexte RAGAS"""
    md = doc.metadata
    return "\n".join([
        f"Titre: {md.get('title', '')}",
        f"Début: {md.get('firstdate_begin', '')}",
        f"Fin: {md.get('lastdate_end', '')}",
        f"Lieu: {md.get('location_name', '')}",
        f"URL: {md.get('canonicalurl', '')}",
    ])


def run(input_csv: Path, out_csv: Path) -> None:
    engine = RAGEngine(RAGConfig(retrieve_k=80, max_events=8))

    # LLM juge : Mistral
    llm = LangchainLLMWrapper(ChatMistralAI(model="mistral-large-latest", temperature=0))

    samples = []
    with input_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            question = r["question"]
            ground_truth = r.get("ground_truth", "")
            # Utilise la réponse pré-générée dans le CSV 
            answer = r.get("answer", "")

            # Récupère les contextes réellement utilisés par le RAG
            retrieved = engine._retrieve(question)
            filtered = engine._filter(retrieved, question)
            contexts = [build_context(d) for d in filtered[: engine.config.max_events]]

            # Si la réponse n'est pas dans le CSV, on la génère
            if not answer:
                answer = engine.ask(question)["answer"]

            samples.append(SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=ground_truth,   # nécessaire pour ContextPrecision et ContextRecall
            ))
            print(f"Question traitée : {question[:60]}...")

    dataset = EvaluationDataset(samples=samples)

    metrics = [
        Faithfulness(llm=llm),       # réponse fidèle au contexte ?
        ContextPrecision(llm=llm),   # contexte récupéré précis (peu de bruit) ?
        ContextRecall(llm=llm),      # infos clés bien récupérées ?
    ]

    print("\nEvaluation RAGAS en cours...")
    results = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False, show_progress=True)

    df = results.to_pandas()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\nRésultats RAGAS")
    print(results)
    print(f"\nRésultats sauvegardés : {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/ref_paris.csv")
    parser.add_argument("--out", default="data/eval/ragas_results.csv")
    args = parser.parse_args()
    run(Path(args.input), Path(args.out))
