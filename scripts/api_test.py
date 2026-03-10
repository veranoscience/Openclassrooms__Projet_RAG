import os
import requests

BASE_URL = os.getenv("API_URL", "http://localhost:8000")

payload = {
    "question": "Quels événements culturels à Paris en 2026 ?",
    "top_k": 80,
    "max_events": 6
}

r = requests.post(f"{BASE_URL}/ask", json=payload, timeout=120)
if r.status_code != 200:
    print("Erreur:", r.status_code, r.text)
    exit(1)

data = r.json()

print("Answer:\n", data["answer"])
print("\nEvents:")
for e in data["events"]:
    print("-", e.get("title"), "|", e.get("start"), "|", e.get("url"))