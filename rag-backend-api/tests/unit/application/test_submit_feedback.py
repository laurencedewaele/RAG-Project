import unittest

from src.application.dto.feedback_dto import SubmitFeedbackCommand
from src.application.use_cases.submit_feedback import SubmitFeedbackService


class FakeObservabilityPort:
    def __init__(self):
        self.scores = []

    def create_score(self, trace_id, name, value, comment=None, observation_id=None):
        self.scores.append((trace_id, name, value, comment))


class SubmitFeedbackServiceTests(unittest.TestCase):
    def test_records_score_and_event(self):
        observability = FakeObservabilityPort()
        service = SubmitFeedbackService(observability_port=observability)

        result = service.execute(
            SubmitFeedbackCommand(
                trace_id="trace-1",
                score_value="OK",
                feedback_type="helpful",
                comment="great",
            )
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(observability.scores[0], ("trace-1", "user_feedback", "OK", "great"))


if __name__ == "__main__":
    unittest.main()
