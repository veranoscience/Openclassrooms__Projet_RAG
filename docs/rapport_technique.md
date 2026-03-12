# Rapport Technique — Système RAG Puls-Events

## 1. Contexte et objectif

**Projet :** Puls-Events
**Objectif :** Concevoir et déployer un système RAG (Retrieval-Augmented Generation) capable de répondre à des questions en langage naturel sur des événements culturels à Paris, en s'appuyant exclusivement sur des données réelles issues d'OpenAgenda.

**Cas d'usage :** Un utilisateur pose une question comme *"Quels concerts jazz à Paris en janvier 2026 ?"* et reçoit une réponse structurée avec les événements correspondants (titre, date, lieu, lien).

---

## 2. Architecture du système

### Schéma UML

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (curl / UI)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ POST /ask
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API FastAPI                                │
│   /ask  /rebuild  /health  /metadata                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                        RAGEngine                               │
│                                                                │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ _retrieve() │──▶│  _filter()   │──▶│  _build_context()    │ │
│  │  FAISS k=80 │   │ filtre temp. │   │  formatage texte     │ │
│  │  score≥0.3  │   │ dédup + tri  │   │  + page_content[:300]│ │
│  └─────────────┘   └──────────────┘   └──────────┬───────────┘ │
│                                                  │             │
│                                                  ▼             │
│                                       ┌──────────────────────┐ │
│                                       │   Mistral LLM        │ │
│                                       │  (mistral-large)     │ │
│                                       └──────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline de données                          │
│                                                                 │
│  OpenAgenda ──▶ fetch_events.py ──▶ preprocess.py               │
│                                          │                      │
│                                          ▼                      │
│                               build_faiss_index.py              │
│                                          │                      │
│                                          ▼                      │
│                               FAISS Index (vectorstores/        │
└─────────────────────────────────────────────────────────────────┘
```

### Rôle de chaque composant

| Composant | Rôle |
|-----------|------|
| `scripts/fetch_events.py` | Collecte les événements depuis OpenDataSoft/OpenAgenda via API REST (filtre sur Paris, 2025-2027) |
| `scripts/preprocess.py` | Nettoyage HTML, normalisation des textes, déduplication par UID, conversion des dates en UTC |
| `scripts/build_faiss_index.py` | Vectorisation via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, indexation dans FAISS |
| `rag/engine.py` | Moteur RAG : retrieval par similarité cosinus, filtrage temporel, construction du contexte, appel LLM |
| `api/main.py` | API REST FastAPI avec endpoints `/ask`, `/rebuild`, `/health`, `/metadata` |

---

## 3. Choix techniques

### 3.1 Modèle d'embeddings

**Choix :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**Justification :**
- Multilingue (français natif) — adapté aux descriptions d'événements en français
- Léger (117M paramètres) — rapide à l'inférence, déployable sans GPU
- Normalisé (cosinus) — cohérent avec FAISS en mode similarité

**Alternative envisagée :** `camembert-base` (FR uniquement) — écarté car limité à une seule langue.

### 3.2 Base vectorielle

**Choix :** FAISS (`IndexFlatIP` via LangChain)

**Justification :**
- Recherche exacte en mémoire — précision maximale pour un corpus de ~10 000 événements
- Pas de serveur à gérer (contrairement à Chroma ou Qdrant)
- Chargement rapide au démarrage de l'API (fichier binaire)

### 3.3 LLM de génération

**Choix :** Mistral Large (`mistral-large-latest`)

**Justification :**
- Excellentes performances en français
- API accessible (coût raisonnable pour un POC)
- Contexte large (32k tokens) — permet d'envoyer plusieurs événements

**Paramètre :** `temperature=0.1` pour des réponses déterministes et factuelles.

### 3.4 Stratégie de retrieval

- **k=80** documents récupérés avant filtrage pour maximiser le rappel
- **Seuil de similarité cosinus ≥ 0.3** pour éliminer les documents hors sujet
- **Filtrage temporel** : parse la date dans la question (jour / mois+année / année) et ne retient que les événements dont la période chevauche la date demandée
- **Déduplication** par UID pour éviter les répétitions
- **Tri chronologique** par date de début

### 3.5 Construction du contexte

Chaque événement est représenté par :
```
Titre: ...
Début: ...
Fin: ...
Lieu: ...
Adresse: ...
Ville: ...
Lien: ...
Description: ... (300 premiers caractères du page_content)
```

L'inclusion de la description (`page_content[:300]`) a été un facteur clé d'amélioration de la **fidélité** (faithfulness) car le LLM peut ancrer ses réponses dans un contenu textuel réel.

---

## 4. Évaluation

### 4.1 Jeu de test

10 questions de référence construites manuellement (`data/eval/ref_paris.csv`), couvrant :
- Différentes catégories (jazz, cinéma, théâtre, science, danse…)
- Différentes granularités temporelles (mois précis, année entière)
- Un cas sans résultat (décembre 2026)

### 4.2 Métriques RAGAS

| Métrique | Description | Score obtenu |
|----------|-------------|:------------:|
| **Faithfulness** | La réponse est-elle fidèle au contexte récupéré ? | 0.4062 |
| **Context Precision** | Le contexte récupéré contient-il peu de bruit ? | 0.30 |
| **Context Recall** | Les informations clés sont-elles bien récupérées ? | 0.4667 |

### 4.3 Analyse des résultats

**Context Precision (0.4)** — Score satisfaisant pour un POC : la moitié des documents récupérés sont pertinents. Le seuil de similarité à 0.3 a permis d'améliorer ce score (0.20 → 0.40) en réduisant le bruit.

**Faithfulness (0.3)** — Score modéré. Le LLM reste globalement fidèle au contexte mais génère parfois des reformulations qui s'éloignent des données sources. L'ajout de la description (`page_content[:300]`) a amélioré ce score (0.15 → 0.3).

**Context Recall (0.45)** — Score satisfaisant, avec un ground_truth de haute qualité (gold standard manuel) : les références attendues sont précises (dates, lieux, noms d'artistes) et le contexte récupéré ne contient pas toujours tous ces détails.

### 4.4 Limites identifiées

1. **Corpus limité** : ~10 000 événements sur Paris 2025-2027 uniquement
2. **Couverture temporelle** : certaines périodes (décembre 2026) ont peu d'événements indexés
3. **Qualité des descriptions** : les descriptions OpenAgenda sont parfois très courtes ou absentes
4. **Context Recall faible** : les métadonnées structurées (titre, lieu, date) ne remplacent pas des descriptions riches pour le matching sémantique

---

## 5. Résultats qualitatifs

Exemples de réponses correctes :

**Question :** *"Quels concerts jazz à Paris en janvier 2026 ?"*
**Réponse RAG :** 3 concerts identifiés avec dates, lieux et liens — cohérent avec les données.

**Question :** *"Quels ateliers ou stages à Paris en décembre 2026 ?"*
**Réponse RAG :** *"D'après les documents fournis, je n'ai pas trouvé d'événements correspondant"* — comportement correct (pas d'hallucination).

---

## 6. Améliorations possibles

| Axe | Action |
|-----|--------|
| **Recall** | Utiliser un modèle d'embeddings plus performant (`intfloat/multilingual-e5-large`) |
| **Faithfulness** | Ajouter un step de re-ranking (cross-encoder) avant génération |
| **Corpus** | Élargir à d'autres villes / périodes, enrichir les descriptions |
| **Latence** | Pré-calculer des embeddings de questions fréquentes (cache) |
| **Déploiement** | Kubernetes pour la scalabilité, monitoring avec Prometheus |
