from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.domain.model.rag_models import Document, GenerationResult


class LlmPort(ABC):
    @abstractmethod
    def generate_answer(self, question: str, chunks: List[Document]) -> GenerationResult:
        raise NotImplementedError

    def get_model_name(self) -> str | None:
        return None

    def get_generation_config(self) -> Dict[str, Any]:
        """Return the actual generation config in use (e.g. thinking_budget, max_output_tokens).

        This is infrastructure/env-var driven configuration, exposed here so the
        application layer can report real values to observability without the
        domain model having to carry infrastructure concerns.
        """
        return {}
