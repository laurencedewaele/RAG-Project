# chatbot-rag-frontend

## Présentation

Ce chatbot est un assistant conversationnel de type RAG (*Retrieval-Augmented Generation*) conçu pour répondre aux questions des visiteurs d'un site d'association, uniquement à partir des documents et des pages de l'association. Le projet couvre la chaîne complète : un backend RAG, une interface de chat, et une boucle d'évaluation et d'observabilité pour mesurer et améliorer la qualité des réponses.

### Le problème

Un assistant basé sur un LLM seul invente facilement des réponses. Pour un site d'association, c'est un risque : une information fausse sur un événement, une adhésion ou une activité nuit à la confiance. L'objectif était un assistant **fiable et factuel**, qui s'appuie sur des sources vérifiables et sait reconnaître quand il ne sait pas.

### La solution

- **Recherche sémantique** : la question est vectorisée avec un modèle d'embeddings multilingue (`paraphrase-multilingual-MiniLM-L12-v2`), puis comparée aux passages indexés dans une base vectorielle **ChromaDB**.
- **Diversité des résultats** : un reclassement **MMR** (*Maximal Marginal Relevance*) évite de renvoyer plusieurs passages redondants. Un seuil de similarité écarte les passages hors sujet.
- **Génération contrainte** : un LLM **Gemini** reçoit les passages retenus avec un prompt strict. Il répond uniquement à partir du contexte, sans supposition ni extrapolation, demande une précision si la question est floue, et rappelle son périmètre quand la réponse est absente des documents.
- **Transparence** : chaque réponse s'affiche avec les passages sources, leur titre et leur score de similarité.
- **Boucle d'amélioration** : l'utilisateur note chaque réponse (👍 / 👎) et peut laisser un commentaire. Les requêtes, réponses et retours sont tracés avec **Langfuse**, et le backend intègre **Ragas** pour évaluer automatiquement la qualité des réponses.

### Une interface pensée pour expérimenter

L'interface **Gradio** sert aussi de banc d'essai : le nombre de passages (*top k*), l'activation du MMR et son équilibre pertinence/diversité, ainsi que le seuil de similarité se règlent en direct. On voit tout de suite l'effet de chaque paramètre sur les sources retrouvées et sur la réponse.

### Architecture et choix techniques

- **Backend en architecture hexagonale** (FastAPI) : domaine, cas d'usage, ports et adaptateurs sont séparés. Le moteur d'embeddings, la base vectorielle, le LLM et l'outil d'observabilité sont interchangeables sans toucher à la logique métier.
- **Frontend découplé** : l'interface ne communique avec le backend que par son API REST (`/api/v1/ask` et `/api/v1/feedback`). Elle se déploie séparément, sur son propre **Hugging Face Space**, comme le backend.
- **Tests** : tests unitaires des cas d'usage et tests de contrat HTTP (pytest).
- **Données** : la base vectorielle est publiée comme dataset Hugging Face et téléchargée au démarrage du backend.

### Stack

Python 3.12 · FastAPI · Gradio · ChromaDB · Sentence-Transformers · LangChain · Gemini (`google-genai`) · Langfuse · Ragas · Hugging Face (Spaces, Hub) · Docker · pytest

## Hugging Face

- Space (dépôt Git `origin`) : https://huggingface.co/spaces/Loren/rag-frontend
- Application en ligne : https://loren-rag-frontend.hf.space
- Le Space est de type `sdk: gradio` (voir le front-matter de [README.md](README.md)) : il installe [requirements.txt](requirements.txt) et lance `python app.py` (`app_file: app.py`) à chaque `git push` vers `origin`.
- Le frontend appelle le backend, hébergé lui aussi sur un Space Hugging Face ([rag-backend-api](https://huggingface.co/spaces/Loren/rag-backend-api), API sur https://loren-rag-backend-api.hf.space). Définir la variable `BACKEND_URL` du Space vers cette URL, dans *Settings > Variables and secrets*. Sans cela, l'application cherche le backend sur `http://127.0.0.1:8000`.
- Les autres variables de la section ci-dessous (`DEFAULT_TOP_K`, `DEFAULT_SIMILARITY_THRESHOLD`, etc.) sont optionnelles et se définissent au même endroit. Le frontend ne contient aucun secret : les clés (Gemini, Langfuse, token Hugging Face) restent côté backend.

## Exécution locale

Depuis la racine du workspace, placez-vous dans le dossier `rag-frontend`. Le backend RAG doit être démarré séparément et accessible à l'adresse indiquée par `BACKEND_URL`.

```bash
cd rag-frontend
python -m venv .venv
```

Activez l'environnement virtuel selon votre système :

```bash
# Linux / macOS
source .venv/bin/activate

# Windows avec Git Bash
source .venv/Scripts/activate
```

Installez les dépendances, préparez la configuration locale et démarrez l'interface :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python app.py
```

L'interface est disponible par défaut sur `http://127.0.0.1:7860`. Si ce port est occupé, l'application essaie les ports suivants. Modifiez `BACKEND_URL` dans `.env` si le backend n'est pas accessible sur `http://127.0.0.1:8000`.

## Variables d'environnement

Les valeurs suivantes sont utilisées lorsque la variable correspondante n'est pas définie dans `.env` ou dans l'environnement du processus :

| Variable | Valeur par défaut | Rôle |
| --- | --- | --- |
| `BACKEND_URL` | `http://127.0.0.1:8000` | URL de base de l'API RAG |
| `DEFAULT_TOP_K` | `3` | Nombre de passages à rechercher |
| `DEFAULT_USE_MMR` | `true` | Active le reclassement MMR |
| `DEFAULT_LAMBDA_MULT` | `0.55` | Équilibre diversité et pertinence du MMR |
| `DEFAULT_SIMILARITY_THRESHOLD` | `0.4` | Seuil minimal de similarité |
| `DEFAULT_TRACE_NAME` | `hf-space-chat` | Nom de trace envoyé au backend |
| `DEFAULT_USER_ID` | `anonymous` | Identifiant utilisateur envoyé au backend |
| `DEFAULT_BUDGET` | `200` | Valeur initiale du curseur de budget de réflexion |
| `PORT` | `GRADIO_SERVER_PORT`, sinon `7860` | Premier port essayé pour Gradio |

`DEFAULT_TOP_K`, `DEFAULT_USE_MMR`, `DEFAULT_LAMBDA_MULT` et `DEFAULT_SIMILARITY_THRESHOLD` initialisent les réglages de recherche de l'interface. `DEFAULT_BUDGET` initialise le curseur correspondant.


