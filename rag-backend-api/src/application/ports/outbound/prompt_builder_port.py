from abc import ABC, abstractmethod
from typing import List

from src.domain.model.rag_models import Document


class PromptBuilderPort(ABC):
    @abstractmethod
    def build_prompt(self, question: str, chunks: List[Document]) -> str:
        raise NotImplementedError