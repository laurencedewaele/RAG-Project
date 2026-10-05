from abc import ABC, abstractmethod

from src.application.dto.ask_question_dto import AskQuestionCommand, AskQuestionResult


class AskQuestionUseCase(ABC):
    @abstractmethod
    def execute(self, command: AskQuestionCommand) -> AskQuestionResult:
        raise NotImplementedError
