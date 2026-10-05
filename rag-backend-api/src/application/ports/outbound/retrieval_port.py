from abc import ABC, abstractmethod
from typing import List, Optional

from src.domain.model.rag_models import Document


class RetrievalPort(ABC):
    @abstractmethod
    def retrieve(
        self,
        question: str,
        top_k: int,
        lambda_mult: Optional[float],
        similarity_threshold: float,
    ) -> List[Document]:
        raise NotImplementedError
