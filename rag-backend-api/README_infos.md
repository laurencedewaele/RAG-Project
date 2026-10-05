# rag-backend-api

## Présentation

**rag-backend-api** est une API de *Retrieval-Augmented Generation* (RAG) qui répond à des questions en langage naturel à partir d'un corpus littéraire indexé dans ChromaDB. À chaque question, l'API retrouve les passages les plus pertinents (embeddings multilingues + recherche MMR avec seuil de similarité), les injecte dans un prompt, puis demande à Gemini de formuler une réponse fondée sur ces sources. Elle renvoie la réponse, les extraits utilisés (titre, auteur, score) et un `trace_id`.

Points forts du projet :

- **Architecture hexagonale** (domaine / application / adapters / bootstrap) : LLM, vector store, embeddings, observabilité et évaluation sont des ports interchangeables.
- **Stack** : FastAPI, ChromaDB, Sentence-Transformers (HuggingFace), Google Gemini, Docker.
- **Observabilité LLM** avec Langfuse (traces, latences, tokens, coûts, feedback utilisateur), avec repli automatique sur des logs locaux.
- **Évaluation automatique des réponses** avec RAGAS, scores remontés dans Langfuse.
- **Déploiement continu** sur un Space Hugging Face (Docker), base vectorielle chargée depuis un dataset Hugging Face.
- Tests unitaires et de contrat (`pytest`).

## Hugging Face

- Space (dépôt Git `origin`) : https://huggingface.co/spaces/Loren/rag-backend-api
- Le Space est de type `sdk: docker` (voir le front-matter de [README.md](README.md)) : il construit le [Dockerfile](Dockerfile) et lance `uvicorn` à chaque `git push` vers `origin`.
- Les variables d'environnement et les secrets (`GOOGLE_API_KEY`, `API_HF_TOKEN`, clés Langfuse, etc.) se définissent dans *Settings > Variables and secrets* du Space (les clés en **secrets**).
- La base ChromaDB n'est pas dans le repo : elle est téléchargée au démarrage depuis le dataset HF `{HF_ORG_NAME}/{HF_DATASET_NAME}` (par défaut `guild-open-tech/chroma-data`) avec le token `API_HF_TOKEN`.

## URLs

| Swagger (OpenAPI) | Santé |
|---|---|
| https://loren-rag-backend-api.hf.space/docs | https://loren-rag-backend-api.hf.space/health |

La racine `https://loren-rag-backend-api.hf.space/` affiche un message d'accueil JSON avec les liens vers `/docs` et `/health`.

Schéma OpenAPI brut : `/openapi.json`. Redoc : `/redoc`.

Endpoints :

- `GET /` : message d'accueil et liens utiles.
- `POST /api/v1/ask` : pose une question (`question`, `top_k`, `use_mmr`, `lambda_mult`, `similarity_threshold`, `trace_name`, `user_id`).
- `POST /api/v1/feedback` : envoie un feedback sur une réponse (`trace_id`, `score_value`, `feedback_type`, `comment`).
- `GET /health` : état du service.


## Variables d'environnement

### LLM (Gemini)

| Variable | Défaut | Usage |
|---|---|---|
| `GOOGLE_API_KEY` | (aucun) | **Obligatoire.** Clé API Google lue par le SDK `google-genai` pour appeler Gemini. |
| `MODEL_NAME` | `gemini-2.5-flash` | Modèle de génération. Le coût Langfuse n'est calculé que pour les modèles connus de `MODEL_PRICING_PER_1K` dans [ask_question.py](src/application/use_cases/ask_question.py). |
| `LLM_TEMPERATURE` | `0.1` | Température de génération. |
| `LLM_MAX_OUTPUT_TOKENS` | `512` | Nombre maximal de tokens de la réponse. |
| `LLM_THINKING_BUDGET` | `200` | Budget de tokens de « réflexion » de Gemini. |
| `LLM_SEED` | `42` | Seed pour des réponses reproductibles. |
| `DEFAULT_ANSWER` | `Je ne dispose pas d'informations sur ce sujet.` | Réponse renvoyée quand aucun extrait ne dépasse le seuil de similarité (le LLM n'est alors pas appelé). |

### Embeddings

| Variable | Défaut | Usage |
|---|---|---|
| `EMBED_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modèle d'embedding des questions. Doit être le même que celui utilisé pour indexer la collection Chroma. Absent de `.env.example` mais pris en compte. Réutilisé par RAGAS. |

### Base vectorielle (ChromaDB / dataset Hugging Face)

| Variable | Défaut | Usage |
|---|---|---|
| `API_HF_TOKEN` | (aucun) | **Obligatoire.** Token Hugging Face pour télécharger la base Chroma depuis le dataset. |
| `HF_ORG_NAME` | `guild-open-tech` | Organisation / utilisateur propriétaire du dataset HF. |
| `HF_DATASET_NAME` | `chroma-data` | Nom du dataset HF contenant la base Chroma. |
| `COLLECTION_NAME` | `miss_terry` | Collection Chroma interrogée. |
| `HF_CACHE_DIR` | `/tmp` | Dossier de cache HF (`HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` en dérivent). **En local sous Windows, `/tmp` n'existe pas : définir par exemple `HF_CACHE_DIR=./.cache/hf`.** |

### Observabilité (Langfuse)

Langfuse est activé uniquement si `LANGFUSE_PUBLIC_KEY` **et** `LANGFUSE_SECRET_KEY` sont renseignées. Sinon l'API utilise l'observabilité locale par logs.

| Variable | Défaut | Usage |
|---|---|---|
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` (`.env.example`) | URL de l'instance Langfuse (lue par le SDK). |
| `LANGFUSE_PUBLIC_KEY` | (vide) | Clé publique du projet Langfuse. |
| `LANGFUSE_SECRET_KEY` | (vide) | Clé secrète du projet Langfuse. |

### Évaluation RAGAS

| Variable | Défaut | Usage |
|---|---|---|
| `ENABLE_RAGAS_EVAL` | `true` | `true` active l'évaluation RAGAS des réponses ; toute autre valeur la désactive. |
| `RAGAS_LLM_MODEL_NAME` | `gpt-4o-mini` | Modèle LLM juge utilisé par RAGAS. |
| `OPENAI_API_KEY` | (vide) | Clé OpenAI nécessaire au LLM juge. Sans elle, l'initialisation de RAGAS échoue, un warning est loggé (`ragas.adapter.init_failed`) et l'API continue sans évaluation. |

Les embeddings utilisés par RAGAS sont ceux de `EMBED_MODEL_NAME`.

### Logs et divers

| Variable | Défaut | Usage |
|---|---|---|
| `LOG_FILE` | `./logs/app.log` | Fichier de logs, rotation quotidienne à minuit, 15 jours conservés. Le dossier doit exister. |
| `API_URL` | (aucun) | Optionnelle, uniquement recopiée dans les métadonnées Langfuse de l'étape de retrieval. |

## Observabilité (Langfuse)

Les traces `rag-pipeline` contiennent les étapes `retrieval` et `answer-generation` (latence, tokens, coût estimé). Les feedbacks envoyés via `POST /api/v1/feedback` sont rattachés au `trace_id` renvoyé par `/api/v1/ask`. Sans clés Langfuse, les mêmes événements sont écrits dans `logs/app.log`.

## Évaluation RAGAS

Après chaque réponse, RAGAS calcule des métriques de qualité (LLM juge `RAGAS_LLM_MODEL_NAME` et embeddings `EMBED_MODEL_NAME`) publiées comme scores Langfuse. Désactivable avec `ENABLE_RAGAS_EVAL=false`.

---
