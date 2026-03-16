"""
Evaluation baseline (offline, sans LLM, sans API)
Mesure : latence, nb_events retournés, réponse non vide

"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd

from rag.engine import RAGEngine, RAGConfig


def run(input_csv: Path, out_csv: Path) -> None:
    engine = RAGEngine(RAGConfig(retrieve_k=80, max_events=8))

    rows = []
    with input_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            question = r["question"]

            t0 = time.time()
            result = engine.ask(question)
            latency = round(time.time() - t0, 3)

            answer = result["answer"]
            nb_events = len(result["events"])

            rows.append({
                "id": r["id"],
                "question": question,
                "nb_events": nb_events,
                "latency_s": latency,
                "answer_non_vide": len(answer.strip()) > 0,
                "answer": answer[:200],  # aperçu
            })

            print(f"[{r['id']}] {nb_events} events | {latency}s | {answer[:80]}...")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n Résumé ")
    print(f"Questions testées : {len(df)}")
    print(f"Latence moyenne   : {df['latency_s'].mean():.2f}s")
    print(f"Events moyens     : {df['nb_events'].mean():.1f}")
    print(f"Réponses non vides: {df['answer_non_vide'].sum()}/{len(df)}")
    print(f"\nRésultats sauvegardés : {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/ref_paris.csv")
    parser.add_argument("--out", default="data/eval/baseline_results.csv")
    args = parser.parse_args()
    run(Path(args.input), Path(args.out))
