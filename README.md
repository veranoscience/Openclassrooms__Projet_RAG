# Puls-Events RAG

Assistant intelligent de recommandation d'événements culturels à Paris.
Architecture **RAG** (Retrieval-Augmented Generation) : FAISS + LangChain + Mistral, exposée via une API FastAPI

---

## Architecture

```
Requête utilisateur
       │
       ▼
  API FastAPI (/ask)
       │
       ▼
  RAGEngine
  ├── _retrieve()  →  FAISS (similarité cosinus, seuil 0.3)
  ├── _filter()    →  Filtre temporel + déduplication + tri
  ├── _build_context()  →  Formatage du contexte
  └── LLM Mistral  →  Génération de la réponse
```

**Composants :**
| Composant | Rôle |
|-----------|------|
| `scripts/fetch_events.py` | Collecte les événements via OpenDataSoft / OpenAgenda |
| `scripts/preprocess.py` | Nettoyage HTML, normalisation, déduplication |
| `scripts/build_faiss_index.py` | Vectorisation et indexation FAISS |
| `rag/engine.py` | Cœur du RAG : retrieval, filtrage, génération |
| `api/main.py` | API REST FastAPI |

---

## Structure du projet

```
projet_rag/
├── api/                  # API FastAPI (main.py, schemas.py, rebuild.py)
├── rag/                  # Moteur RAG (engine.py)
├── scripts/              # Collecte, preprocessing, indexation, évaluation
├── tests/                # Tests unitaires (pytest)
├── data/
│   ├── processed/        # CSV nettoyé
│   └── eval/             # Jeu de test annoté + résultats RAGAS
├── vectorstores/         # Index FAISS
├── docs/                 # Rapport technique
├── .github/workflows/    # CI GitHub Actions
├── requirements.txt
└── .env                  # Variables d'environnement (non versionné)
```

---

## Prérequis

- Python 3.12+
- Clé API Mistral ([console.mistral.ai](https://console.mistral.ai))
- (Optionnel) Docker

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/veranoscience/projet-rag.git
cd projet-rag

# 2. Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Configuration

Créer un fichier `.env` à la racine :

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

Variables optionnelles :

```env
RETRIEVE_K=80          # Nombre de documents récupérés avant filtrage
MAX_EVENTS=8           # Nombre d'événements max retournés
REBUILD_TOKEN=secret   # Token pour protéger l'endpoint /rebuild
```

---

## Pipeline de données 

```bash
# 1. Collecter les événements depuis OpenAgenda
python scripts/fetch_events.py

# 2. Nettoyer et normaliser les données
python scripts/preprocess.py

# 3. Construire l'index FAISS
python scripts/build_faiss_index.py --force
```

> L'index FAISS est déjà inclus dans le dépôt (`vectorstores/faiss_index/`).
> Ces étapes sont nécessaires uniquement pour reconstruire à partir de nouvelles données

---

## Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation interactive disponible sur : [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Utilisation

### Vérifier l'état de l'API

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Poser une question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts jazz à Paris en janvier 2026 ?", "top_k": 80, "max_events": 5}'
```

Réponse :
```json
{
  "answer": "Voici des concerts jazz à Paris en janvier 2026 ...",
  "events": [
    {
      "uid": "...",
      "title": "PEGAZZ & l'HELICON : Rose(S)",
      "start": "2026-01-17T19:00:00+00:00",
      "end": "2026-01-17T21:00:00+00:00",
      "location_name": "Le Pan Piper",
      "location_address": "2 Impasse Lamier",
      "city": "Paris",
      "lien": "https://openagenda.com/..."
    }
  ]
}
```

### Reconstruire l'index

```bash
curl -X POST http://localhost:8000/rebuild \
  -H "Content-Type: application/json" \
  -H "x-rebuild-token: secret" \
  -d '{"mode": "index_only", "force": true}'
```

---

## Tests

```bash
# Lancer tous les tests
python -m pytest tests/ -v

# Tests par module
python -m pytest tests/test_preprocess.py   # nettoyage
python -m pytest tests/test_indexation.py   # vectorisation FAISS
python -m pytest tests/test_engine.py       # moteur RAG
python -m pytest tests/test_api.py          # endpoints API
```

Les tests utilisent des mocks pour FAISS et Mistral : aucune clé API ni GPU requis

---

## Évaluation RAGAS

```bash
# Générer le jeu de test (nécessite l'API lancée)
python scripts/generate_references.py

# Lancer l'évaluation (métriques : faithfulness, context_precision, context_recall)
python scripts/eval_ragas.py --input data/eval/ref_paris.csv --out data/eval/ragas_results.csv
```


---

## Docker

```bash
# Construire l'image
docker build -t puls-events-rag .

# Lancer le conteneur
docker run -p 8000:8000 --env-file .env puls-events-rag
```

---

## CI/CD

Un workflow GitHub Actions (`.github/workflows/ci.yml`) lance automatiquement les tests unitaires à chaque push sur `main` ou `dev`.
