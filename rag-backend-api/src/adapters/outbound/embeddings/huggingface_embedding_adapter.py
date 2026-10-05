import os
from typing import Any

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from src.application.ports.outbound.embedding_port import EmbeddingPort


class NormalizedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        vec = np.array(super().embed_query(text))
        return (vec / np.linalg.norm(vec)).tolist()


class HuggingFaceEmbeddingAdapter(EmbeddingPort):
    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv(
            "EMBED_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self._embedding_function = NormalizedHuggingFaceEmbeddings(model_name=self._model_name)
        self._embed_model = SentenceTransformer(self._model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()

    def get_embedding_function(self) -> Any:
        return self._embedding_function

    def get_model_name(self) -> str | None:
        return self._model_name