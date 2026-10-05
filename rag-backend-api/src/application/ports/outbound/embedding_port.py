from abc import ABC, abstractmethod
from typing import Any


class EmbeddingPort(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def get_embedding_function(self) -> Any:
        raise NotImplementedError

    def get_model_name(self) -> str | None:
        return None