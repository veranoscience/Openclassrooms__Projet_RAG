# Puls-Events RAG projet

C'est un projet d'un assistant intelligent capable de répondre à des questions sur des événements culturels à partir de données OpenAgenda,
en utilisant une archtecture RAG (Retrieval-Augmented Generation) avec LangChain, FAISS et Mistral.

## Objectifs du projet:

- Récupérer des événements culturels via OpenAgenda
- Nettoyer et sctructurer les données
- Vectoriser les discriptions d'événements
- Indexer les données dans FAISS
- Générer des réponses augmentées avec Mistral
- Exposer le système via une API FastAPI

## Structure du projet

```text
api/         # API FastAPI
rag/         # logique métier du système RAG
scripts/     # scropts de collecte, preprocessing, indexation, évaluation
tests/       # tests unitaires et d'intégration
data/        # donées brutes et traitées
vectore/     # index FAISS
docs/        # documentatio, rapport, slides
````

## Pré-requis

- Python 3.11 ou supérieur
- VS Code ou terminal
- Environnement virtuel Python

## Installation
1. Cloner le projet
````
git clone <url-du-repo>
cd projet_rag
````
2. Créer l'environnement virtuel
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Installer les dépendances
```
pip install -r requirements.txt
```

4. Configurer les variables d’environnement
```
cp .env .env
```






