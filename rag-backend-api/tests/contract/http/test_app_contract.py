import unittest

from fastapi.testclient import TestClient

from src.adapters.inbound.api.app import create_app


class FakeAskQuestionResult:
    def __init__(self):
        self.answer = "answer"
        self.trace_id = "trace-123"
        self.chunks = []


class FakeAskQuestionService:
    def execute(self, command):
        self.last_command = command
        return FakeAskQuestionResult()


class FakeSubmitFeedbackResult:
    def __init__(self):
        self.status = "ok"


class FakeSubmitFeedbackService:
    def execute(self, command):
        self.last_command = command
        return FakeSubmitFeedbackResult()


class FakeContainer:
    def __init__(self):
        self.ask_question = FakeAskQuestionService()
        self.submit_feedback = FakeSubmitFeedbackService()


class AppContractTests(unittest.TestCase):
    def setUp(self):
        self.container = FakeContainer()
        self.client = TestClient(create_app(container=self.container))

    def test_root_endpoint_returns_welcome_message(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "message": "Bienvenue sur l'API RAG.",
                "docs": "/docs",
                "health": "/health",
            },
        )

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ask_endpoint_returns_answer(self):
        response = self.client.post(
            "/api/v1/ask",
            json={
                "question": "Hello?",
                "top_k": 3,
                "use_mmr": False,
                "lambda_mult": None,
                "similarity_threshold": 0.4,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "answer")
        self.assertIn("x-request-id", response.headers)
        self.assertEqual(self.container.ask_question.last_command.question, "Hello?")

    def test_feedback_endpoint_returns_ok(self):
        response = self.client.post(
            "/api/v1/feedback",
            json={
                "trace_id": "trace-123",
                "score_value": "OK",
                "feedback_type": "helpful",
                "comment": "nice",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(self.container.submit_feedback.last_command.trace_id, "trace-123")


if __name__ == "__main__":
    unittest.main()
