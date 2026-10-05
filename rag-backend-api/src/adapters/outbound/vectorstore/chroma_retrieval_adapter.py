import logging
import os
from functools import lru_cache
from typing import Any, List, Optional

import chromadb
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma

from src.application.ports.outbound.embedding_port import EmbeddingPort
from src.application.ports.outbound.retrieval_port import RetrievalPort
from src.domain.model.rag_models import Document


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_vector_resources() -> tuple[Any, Any, str]:
    load_dotenv()

    cache_dir = os.getenv("HF_CACHE_DIR", "/tmp")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_dir, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))

    hf_token = os.getenv("API_HF_TOKEN")
    hf_dataset_name = os.getenv("HF_DATASET_NAME", "chroma-data")
    hf_org_name = os.getenv("HF_ORG_NAME", "guild-open-tech")
    collection_name = os.getenv("COLLECTION_NAME", "miss_terry")

    repo_id = f"{hf_org_name}/{hf_dataset_name}"
    local_chroma_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    chroma_client = chromadb.PersistentClient(path=local_chroma_path)
    collection = chroma_client.get_collection(collection_name)
    return collection, chroma_client, collection_name


class NativeRetrievalAdapter(RetrievalPort):
    def __init__(self, embedding_port: EmbeddingPort) -> None:
        self._embedding_port = embedding_port
        self._collection, chroma_client, collection_name = _load_vector_resources()
        self._vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=self._embedding_port.get_embedding_function(),
        )

    def retrieve(
        self,
        question: str,
        top_k: int,
        lambda_mult: Optional[float],
        similarity_threshold: float,
    ) -> List[Document]:
        if not question:
            return []

        question_embedding = [self._embedding_port.embed_query(question)]

        if lambda_mult is not None:
            computed_fetch_k = max(top_k * 5, 15)
            results = self._vectorstore.max_marginal_relevance_search(
                query=question,
                k=top_k,
                fetch_k=computed_fetch_k,
                lambda_mult=lambda_mult,
            )

            ids = [doc.id for doc in results]
            metadatas = [doc.metadata for doc in results]
            contents = [doc.page_content for doc in results]

            mmr_embeddings = self._collection.get(ids=ids, include=["embeddings"])
            chunk_embeddings = mmr_embeddings.get("embeddings", [])
            if chunk_embeddings is not None and len(chunk_embeddings) > 0:
                q_vec = np.array(question_embedding[0])
                distances = [float(np.dot(q_vec, np.array(emb))) for emb in chunk_embeddings]
                id_to_distance = dict(zip(mmr_embeddings["ids"], distances))
                distances_ordered = [id_to_distance.get(doc_id) for doc_id in ids]
            else:
                distances_ordered = [None] * len(ids)

            return [
                Document(
                    id=doc_id,
                    content=content,
                    score=distance,
                    title=metadata.get("title"),
                    author=metadata.get("author"),
                )
                for doc_id, content, metadata, distance in zip(ids, contents, metadatas, distances_ordered)
                if distance is not None and distance >= similarity_threshold
            ]

        results = self._collection.query(
            query_embeddings=question_embedding,
            n_results=top_k,
        )

        return [
            Document(
                id=doc_id,
                content=content,
                score=1 - distance if distance is not None else None,
                title=metadata.get("title"),
                author=metadata.get("author"),
            )
            for doc_id, content, metadata, distance in zip(
                results.get("ids", [[]])[0],
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            )
            if distance is not None and (1 - distance) >= similarity_threshold
        ]
