from typing import Optional

from src.application.ports.outbound.evaluation_port import EvaluationPort
from src.domain.model.evaluation_models import AnswerEvaluation


class NoOpEvaluationAdapter(EvaluationPort):
    def enqueue(
        self,
        evaluation: AnswerEvaluation,
        trace_id: Optional[str],
        observation_id: Optional[str] = None,
    ) -> None:
        return
