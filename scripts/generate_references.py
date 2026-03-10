import requests
import pandas as pd

API_URL = "http://localhost:8000/ask"

questions = [
   
    "Quels événements cinéma ou court métrage à Paris en mars 2026 ?",
    "Quels concerts jazz à Paris en janvier 2026 ?",
    "Quels événements autour du théâtre à Paris en mai 2026 ?",
    "Quels événements scientifiques ou conférences à Paris en avril 2026 ?",
    "Quels ateliers ou stages à Paris en décembre 2026 ?",
    "Je cherche des événements à Paris sur l’art contemporain en 2026",
    "Quels événements musicaux ou concerts à Paris en 2025 ?",
    "Quels événements de danse ou spectacle vivant à Paris en juin 2026 ?",
    "Quelles visites guidées ou balades culturelles à Paris en janvier 2026 ?",
    "Quels festivals culturels à Paris en mars 2026 ?",
]

rows = []

for i,q in enumerate(questions):

    r = requests.post(API_URL,json={"question":q, "top_k": 200, "max_events": 3})
    data = r.json()

    events = data.get("events",[])[:3]

    # ground_truth : titres des événements retournés (vérité terrain silver)
    ground_truth = "; ".join([e["title"] for e in events if e.get("title")])

    # answer : réponse générée par le RAG
    answer = data.get("answer", "")

    rows.append({
        "id": f"q{i+1}",
        "question": q,
        "answer": answer,
        "contexts": "",        # placeholder, rempli dynamiquement par eval_ragas.py
        "ground_truth": ground_truth,
    })

    print(f"[q{i+1}] ground_truth: {ground_truth[:80] or '(vide)'}")

df = pd.DataFrame(rows)

df.to_csv("data/eval/ref_paris.csv",index=False)

print("\nDataset généré → data/eval/ref_paris.csv")