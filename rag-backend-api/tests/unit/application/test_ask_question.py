import unittest

from src.application.dto.ask_question_dto import AskQuestionCommand
from src.application.ports.outbound.observability_port import ScoreInput
from src.application.use_cases.ask_question import AskQuestionService
from src.domain.errors.exceptions import EmptyQuestionError
from src.domain.model.evaluation_models import AnswerEvaluation
from src.domain.model.rag_models import Document, GenerationResult


class FakeRetrievalPort:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, question, top_k, lambda_mult, similarity_threshold):
        self.calls.append((question, top_k, lambda_mult, similarity_threshold))
        return self.chunks


class FakeLlmPort:
    def __init__(self):
        self.calls = []

    def generate_answer(self, question, chunks):
        self.calls.append((question, chunks))
        return GenerationResult(
            text="answer-from-llm",
            prompt="prompt-from-llm",
            input_tokens=10,
            output_tokens=12,
            thoughts_tokens=3,
            finish_reason="STOP",
        )

    def get_model_name(self):
        return "gemini-2.5-flash"

    def get_generation_config(self):
        return {"thinking_budget": 200, "max_output_tokens": 512}


class FakeObservationHandle:
    def __init__(self, trace_id, span_id, name):
        self.trace_id = trace_id
        self.id = span_id
        self.name = name
        self.updates = []
        self.scores = []
        self.closed = False

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def score(self, name, value=None, data_type=None):
        if isinstance(name, dict):
            # Batch mode: iterate over scores dictionary
            for score_name, score_input in name.items():
                self.scores.append(score_input)
        else:
            # Single mode
            self.scores.append(ScoreInput(name=name, value=value, data_type=data_type))

    def close(self):
        self.closed = True


class FakeObservabilityPort:
    def __init__(self):
        self.traces = []
        self.scores = []
        self.trace_handle = None
        self.all_handles = []

    def start_trace(self, name, payload, *, trace_name=None, user_id=None, session_id=None):
        self.traces.append((name, payload, trace_name, user_id, session_id))
        self.trace_handle = FakeObservationHandle("trace-123", "root-123", name)
        self.all_handles.append(self.trace_handle)
        return self.trace_handle

    def start_observation(self, name, as_type=None, model=None):
        handle = FakeObservationHandle("trace-123", f"{name}-123", name)
        self.all_handles.append(handle)
        return handle

    def finish_trace(self):
        pass

    def create_score(self, trace_id, name, value, comment=None, observation_id=None):
        self.scores.append((trace_id, name, value, comment, observation_id))


class FakeEvaluationPort:
    def __init__(self):
        self.items = []

    def enqueue(self, evaluation, trace_id, observation_id=None):
        self.items.append((evaluation, trace_id, observation_id))


class AskQuestionServiceTests(unittest.TestCase):
    def test_raises_on_empty_question(self):
        service = AskQuestionService(
            retrieval_port=FakeRetrievalPort([]),
            llm_port=FakeLlmPort(),
            observability_port=FakeObservabilityPort(),
            default_answer="fallback",
            evaluation_port=FakeEvaluationPort(),
        )

        with self.assertRaises(EmptyQuestionError):
            service.execute(AskQuestionCommand(question="   "))

    def test_returns_default_answer_when_no_chunks(self):
        observability = FakeObservabilityPort()
        evaluation = FakeEvaluationPort()
        service = AskQuestionService(
            retrieval_port=FakeRetrievalPort([]),
            llm_port=FakeLlmPort(),
            observability_port=observability,
            default_answer="fallback",
            evaluation_port=evaluation,
        )

        result = service.execute(AskQuestionCommand(question="hello"))

        self.assertEqual(result.answer, "fallback")
        self.assertEqual(result.trace_id, "trace-123")
        self.assertEqual(len(result.chunks), 0)
        self.assertEqual(observability.traces[0][0], "rag-pipeline")
        self.assertEqual(observability.traces[0][1]["query"], "hello")
        self.assertTrue(any(update.get("status_message") == "DEFAULT_ANSWER" for handle in observability.all_handles for update in handle.updates))
        self.assertIsInstance(evaluation.items[0][0], AnswerEvaluation)
        self.assertEqual(evaluation.items[0][0].contexts, [])
        self.assertEqual(evaluation.items[0][1], "trace-123")

    def test_uses_llm_when_chunks_exist(self):
        chunk = Document(id="1", content="context", score=0.91)
        retrieval = FakeRetrievalPort([chunk])
        llm = FakeLlmPort()
        observability = FakeObservabilityPort()
        evaluation = FakeEvaluationPort()
        service = AskQuestionService(
            retrieval_port=retrieval,
            llm_port=llm,
            observability_port=observability,
            default_answer="fallback",
            evaluation_port=evaluation,
        )

        result = service.execute(AskQuestionCommand(question="hello", use_mmr=True, lambda_mult=0.5))

        self.assertEqual(result.answer, "answer-from-llm")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(retrieval.calls[0][2], 0.5)
        self.assertEqual(observability.traces[0][1]["query"], "hello")
        self.assertEqual(observability.traces[0][1]["lambda_mult"], 0.5)
        self.assertEqual(llm.calls[0][0], "hello")
        self.assertEqual(observability.all_handles[1].updates[0]["status_message"], "SUCCESS")
        generation_metadata = observability.all_handles[2].updates[0]["metadata"]
        self.assertEqual(generation_metadata["thinking_budget"], 200)
        self.assertEqual(generation_metadata["max_output_tokens"], 512)
        self.assertEqual(evaluation.items[0][0].question, "hello")
        self.assertEqual(evaluation.items[0][0].contexts, ["context"])

    def test_score_payloads_are_emitted_as_score_input_objects(self):
        observability = FakeObservabilityPort()
        evaluation = FakeEvaluationPort()
        service = AskQuestionService(
            retrieval_port=FakeRetrievalPort([Document(id="1", content="context", score=0.91)]),
            llm_port=FakeLlmPort(),
            observability_port=observability,
            default_answer="fallback",
            evaluation_port=evaluation,
        )

        service.execute(AskQuestionCommand(question="hello"))

        self.assertTrue(observability.all_handles[1].scores)
        self.assertIsInstance(observability.all_handles[1].scores[0], ScoreInput)
        self.assertEqual(observability.all_handles[1].scores[0].name, "api_latency")

    def test_trace_payload_uses_none_lambda_mult_when_mmr_disabled(self):
        observability = FakeObservabilityPort()
        evaluation = FakeEvaluationPort()
        service = AskQuestionService(
            retrieval_port=FakeRetrievalPort([]),
            llm_port=FakeLlmPort(),
            observability_port=observability,
            default_answer="fallback",
            evaluation_port=evaluation,
        )

        service.execute(AskQuestionCommand(question="hello", use_mmr=False, lambda_mult=0.5))

        self.assertIsNone(observability.traces[0][1]["lambda_mult"])

    def test_passes_trace_name_and_user_id_and_resets_session_id_per_call(self):
        observability = FakeObservabilityPort()
        evaluation = FakeEvaluationPort()
        service = AskQuestionService(
            retrieval_port=FakeRetrievalPort([]),
            llm_port=FakeLlmPort(),
            observability_port=observability,
            default_answer="fallback",
            evaluation_port=evaluation,
        )

        service.execute(AskQuestionCommand(question="hello", trace_name="chatbot-search", user_id="user-42"))
        service.execute(AskQuestionCommand(question="hello again", trace_name="chatbot-search", user_id="user-42"))

        first_trace_name, first_user_id, first_session_id = observability.traces[0][2:5]
        second_trace_name, second_user_id, second_session_id = observability.traces[1][2:5]

        self.assertEqual(first_trace_name, "chatbot-search")
        self.assertEqual(first_user_id, "user-42")
        self.assertEqual(second_trace_name, "chatbot-search")
        self.assertEqual(second_user_id, "user-42")
        self.assertIsNotNone(first_session_id)
        self.assertIsNotNone(second_session_id)
        self.assertNotEqual(first_session_id, second_session_id)


if __name__ == "__main__":
    unittest.main()
