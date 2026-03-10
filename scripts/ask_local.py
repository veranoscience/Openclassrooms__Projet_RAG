import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.engine import RAGEngine

engine = RAGEngine()

q = "Quelles sont les événements culturels à Paris en mars 2025 ?"
out = engine.ask(q)

print("QUESTION:", q)
print("\nRÉPONSE:\n", out["answer"])
print("\nÉVÉNEMENTS (payload):")
for e in out["events"]:
    print("-", e["title"], "|", e["start"], "|", e["lien"])