from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerEvaluation:
    question: str
    answer: str
    contexts: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    scores: dict[str, float]