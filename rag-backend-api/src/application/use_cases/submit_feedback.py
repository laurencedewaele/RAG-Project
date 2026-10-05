from src.application.dto.feedback_dto import SubmitFeedbackCommand, SubmitFeedbackResult
from src.application.ports.inbound.submit_feedback_use_case import SubmitFeedbackUseCase
from src.application.ports.outbound.observability_port import ObservabilityPort
from src.domain.model.feedback_models import UserFeedback


class SubmitFeedbackService(SubmitFeedbackUseCase):
    def __init__(self, observability_port: ObservabilityPort) -> None:
        self._obs = observability_port

    def execute(self, command: SubmitFeedbackCommand) -> SubmitFeedbackResult:
        feedback = UserFeedback(
            feedback_type=command.feedback_type,
            feedback_comment=command.comment,
        )

        self._obs.create_score(
            trace_id=command.trace_id,
            name="user_feedback",
            value=command.score_value,
            comment=feedback.feedback_comment or f"User {feedback.feedback_type} feedback",
        )
        return SubmitFeedbackResult(status="ok")