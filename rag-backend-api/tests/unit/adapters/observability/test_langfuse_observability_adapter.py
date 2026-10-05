import unittest
from contextlib import contextmanager
from unittest.mock import patch

from src.adapters.outbound.observability import langfuse_observability_adapter as module
from src.adapters.outbound.observability.langfuse_observability_adapter import LangfuseObservabilityAdapter
from src.application.ports.outbound.observability_port import ScoreInput


class FakeLangfuseObservation:
    def __init__(self, trace_id, observation_id):
        self.trace_id = trace_id
        self.id = observation_id
        self.updates = []
        self.scores = []

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def score(self, **kwargs):
        self.scores.append(kwargs)


class FakeLangfuseClient:
    """Mimics the subset of the real Langfuse client used by the adapter."""

    def __init__(self):
        self.created_scores = []
        self.flushed = False
        self._counter = 0

    @contextmanager
    def start_as_current_observation(self, name, as_type=None, model=None):
        self._counter += 1
        yield FakeLangfuseObservation(trace_id="trace-fixed", observation_id=f"obs-{self._counter}")

    def create_score(self, **kwargs):
        self.created_scores.append(kwargs)

    def flush(self):
        self.flushed = True


class FakePropagateAttributes:
    """Records propagate_attributes(...) calls without touching real OTel context."""

    def __init__(self):
        self.calls = []

    @contextmanager
    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        yield


class LangfuseObservabilityAdapterTests(unittest.TestCase):
    def setUp(self):
        self.fake_client = FakeLangfuseClient()
        self.fake_propagate = FakePropagateAttributes()

        get_client_patcher = patch.object(module, "get_client", return_value=self.fake_client)
        propagate_patcher = patch.object(module, "propagate_attributes", self.fake_propagate)
        get_client_patcher.start()
        propagate_patcher.start()
        self.addCleanup(get_client_patcher.stop)
        self.addCleanup(propagate_patcher.stop)

        self.adapter = LangfuseObservabilityAdapter()

    def test_start_trace_returns_handle_and_propagates_attributes(self):
        trace_handle = self.adapter.start_trace(
            name="rag-pipeline",
            payload={"query": "hello"},
            trace_name="chatbot-search",
            user_id="user-42",
            session_id="session-1",
        )

        self.assertIsNotNone(trace_handle)
        self.assertEqual(trace_handle.trace_id, "trace-fixed")
        self.assertEqual(
            self.fake_propagate.calls[-1],
            {"trace_name": "chatbot-search", "user_id": "user-42", "session_id": "session-1"},
        )

    def test_handle_score_scores_the_trace_without_raising(self):
        # Regression test: confirm that handle.score() works correctly on root trace.
        trace_handle = self.adapter.start_trace(name="rag-pipeline", payload={})

        trace_handle.score({"total_latency": ScoreInput(name="total_latency", value=1.23)})

        self.assertEqual(trace_handle.observation.scores, [{"name": "total_latency", "value": 1.23}])

    def test_handle_score_is_safe_with_none_handle(self):
        # Should not raise even though handle is None.
        self.adapter.start_trace(name="rag-pipeline", payload={})
        # None handle should not break score() calls
        none_handle = None
        if none_handle is not None:
            none_handle.score({"total_latency": ScoreInput(name="total_latency", value=1.23)})

    def test_handle_score_scores_trace_even_with_open_child_spans(self):
        trace_handle = self.adapter.start_trace(name="rag-pipeline", payload={})
        retrieval_handle = self.adapter.start_observation(name="retrieval", as_type="tool")

        trace_handle.score({"total_latency": ScoreInput(name="total_latency", value=4.56)})

        self.assertEqual(trace_handle.observation.scores, [{"name": "total_latency", "value": 4.56}])
        self.assertEqual(retrieval_handle.observation.scores, [])

    def test_finish_trace_closes_spans_and_exits_propagation_context(self):
        self.adapter.start_trace(
            name="rag-pipeline",
            payload={},
            trace_name="chatbot-search",
            user_id="user-42",
            session_id="session-1",
        )
        self.adapter.start_observation(name="retrieval", as_type="tool")

        self.adapter.finish_trace()

        self.assertEqual(self.adapter._current_trace.get(), [])
        self.assertEqual(self.adapter._current_span.get(), [])


if __name__ == "__main__":
    unittest.main()
