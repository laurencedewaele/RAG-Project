from abc import ABC, abstractmethod
from typing import Optional

from src.domain.model.evaluation_models import AnswerEvaluation


class EvaluationPort(ABC):
    @abstractmethod
    def enqueue(self, evaluation: AnswerEvaluation, trace_id: Optional[str], observation_id: Optional[str] = None) -> None:
        raise NotImplementedError
