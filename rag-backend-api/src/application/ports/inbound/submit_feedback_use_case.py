from abc import ABC, abstractmethod

from src.application.dto.feedback_dto import SubmitFeedbackCommand, SubmitFeedbackResult


class SubmitFeedbackUseCase(ABC):
    @abstractmethod
    def execute(self, command: SubmitFeedbackCommand) -> SubmitFeedbackResult:
        raise NotImplementedError