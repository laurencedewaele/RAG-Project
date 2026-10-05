import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.application.ports.outbound.observability_port import ObservationHandle, ObservabilityPort, ScoreInput


@dataclass
class _LoggingObservationHandle:
    adapter: "LoggingObservabilityAdapter"
    logger: logging.Logger
    trace_id: str
    observation_id: str
    name: str

    def update(
        self,
        *,
        input: Any | None = None,
        output: Any | None = None,
        status_message: str | None = None,
        metadata: Dict[str, Any] | None = None,
        usage_details: Dict[str, Any] | None = None,
    ) -> None:
        self.logger.info(
            "span.update",
            extra={
                "trace_id": self.trace_id,
                "span_id": self.observation_id,
                "span_name": self.name,
                "input": input,
                "output": output,
                "status_message": status_message,
                "metadata": metadata,
                "usage_details": usage_details,
            },
        )

    def score(self, scores: Dict[str, ScoreInput]) -> None:
        for score_name, score_data in scores.items():
            value = score_data.value if score_data is not None else None
            data_type = score_data.data_type if score_data is not None else None
            self.logger.info(
                "span.score",
                extra={"trace_id": self.trace_id, "span_id": self.observation_id, "score_name": score_name, "value": value, "data_type": data_type},
            )

    def close(self) -> None:
        self.adapter._pop_handle(self)


class LoggingObservabilityAdapter(ObservabilityPort):
    def __init__(self, logger_name: str = "chatbot.backend") -> None:
        self._logger = logging.getLogger(logger_name)
        self._current_trace: ContextVar[list[_LoggingObservationHandle]] = ContextVar("current_trace", default=None)
        self._current_span: ContextVar[list[_LoggingObservationHandle]] = ContextVar("current_span", default=None)

    def _push_handle(self, handle: _LoggingObservationHandle) -> None:
        trace_stack = list(self._current_trace.get() or [])
        span_stack = list(self._current_span.get() or [])
        if not trace_stack:
            trace_stack.append(handle)
        span_stack.append(handle)
        self._current_trace.set(trace_stack)
        self._current_span.set(span_stack)

    def _pop_handle(self, handle: _LoggingObservationHandle) -> None:
        trace_stack = [item for item in self._current_trace.get() or [] if item is not handle]
        span_stack = [item for item in self._current_span.get() or [] if item is not handle]
        self._current_trace.set(trace_stack)
        self._current_span.set(span_stack)

    def start_trace(
        self,
        name: str,
        payload: Dict[str, Any],
        *,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[ObservationHandle]:
        trace_id = str(uuid.uuid4())
        handle = _LoggingObservationHandle(adapter=self, logger=self._logger, trace_id=trace_id, observation_id=str(uuid.uuid4()), name=trace_name or name)
        self._push_handle(handle)
        self._logger.info(
            "trace.start",
            extra={
                "trace_id": trace_id,
                "trace_name": trace_name or name,
                "payload": payload,
                "user_id": user_id,
                "session_id": session_id,
            },
        )
        return handle

    def start_observation(
        self,
        name: str,
        as_type: str | None = None,
        model: str | None = None,
    ) -> Optional[ObservationHandle]:
        parent = self._current_trace.get()
        if parent is None:
            trace_id = str(uuid.uuid4())
            parent = _LoggingObservationHandle(adapter=self, logger=self._logger, trace_id=trace_id, observation_id=str(uuid.uuid4()), name="trace")
            self._push_handle(parent)

        handle = _LoggingObservationHandle(adapter=self, logger=self._logger, trace_id=parent.trace_id, observation_id=str(uuid.uuid4()), name=name)
        self._push_handle(handle)
        self._logger.info("span.start", extra={"trace_id": handle.trace_id, "span_id": handle.observation_id, "span_name": name, "as_type": as_type, "model": model})
        return handle

    def finish_trace(self) -> None:
        span_stack = self._current_span.get() or []
        while span_stack:
            handle = span_stack.pop()
            handle.close()
        trace_stack = self._current_trace.get() or []
        while trace_stack:
            handle = trace_stack.pop()
            handle.close()
        self._current_span.set([])
        self._current_trace.set([])

    def create_score(
        self,
        trace_id: str,
        name: str,
        value: float | str,
        comment: str | None = None,
        observation_id: str | None = None,
    ) -> None:
        self._logger.info(
            "score.create",
            extra={"trace_id": trace_id, "observation_id": observation_id, "score_name": name, "value": value, "comment": comment},
        )
