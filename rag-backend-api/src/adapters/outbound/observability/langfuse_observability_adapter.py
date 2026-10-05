import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional
import importlib

from src.application.ports.outbound.observability_port import ObservationHandle, ObservabilityPort, ScoreInput

try:
    _langfuse_module = importlib.import_module("langfuse")
    get_client = _langfuse_module.get_client
    propagate_attributes = _langfuse_module.propagate_attributes
except Exception:  # pragma: no cover - optional dependency in local tooling
    get_client = None
    propagate_attributes = None


@dataclass
class _LangfuseObservationHandle:
    adapter: "LangfuseObservabilityAdapter"
    logger: logging.Logger
    context_manager: Any | None
    observation: Any | None
    trace_id: Optional[str]
    observation_id: Optional[str]
    name: str
    propagation_cm: Any | None = None

    def update(
        self,
        *,
        input: Any | None = None,
        output: Any | None = None,
        status_message: str | None = None,
        metadata: Dict[str, Any] | None = None,
        usage_details: Dict[str, Any] | None = None,
    ) -> None:
        if self.observation is None:
            return

        kwargs = {
            key: value
            for key, value in {
                "input": input,
                "output": output,
                "status_message": status_message,
                "metadata": metadata,
                "usage_details": usage_details,
            }.items()
            if value is not None
        }

        if kwargs:
            self.observation.update(**kwargs)

    def score(self, scores: Dict[str, ScoreInput]) -> None:
        if self.observation is None:
            return

        scorer = getattr(self.observation, "score", None)
        if not callable(scorer):
            return

        for score_name, score_input in scores.items():
            try:
                payload_name = score_input.name if score_input is not None else score_name
                kwargs = {
                    key: value
                    for key, value in {
                        "name": payload_name,
                        "value": score_input.value if score_input is not None else None,
                        "data_type": score_input.data_type if score_input is not None else None,
                        "comment": score_input.comment if score_input is not None else None,
                    }.items()
                    if value is not None
                }
                scorer(**kwargs)
            except Exception as e:
                self.logger.warning(f"⚠️ Error scoring {score_name}: {e}")

    def close(self) -> None:
        if self.propagation_cm is not None:
            try:
                self.propagation_cm.__exit__(None, None, None)
            except Exception:
                pass
            self.propagation_cm = None
        if self.context_manager is not None:
            self.context_manager.__exit__(None, None, None)
        self.adapter._pop_handle(self)


class LangfuseObservabilityAdapter(ObservabilityPort):
    def __init__(self) -> None:
        self._logger = logging.getLogger("chatbot.backend.observability")
        self._client = None
        self._current_trace: ContextVar[list[_LangfuseObservationHandle]] = ContextVar("langfuse_current_trace", default=None)
        self._current_span: ContextVar[list[_LangfuseObservationHandle]] = ContextVar("langfuse_current_span", default=None)
        try:
            if get_client is None:
                raise ImportError("langfuse not installed")
            self._client = get_client()
            self._logger.info("langfuse.client.initialized")
        except Exception as exc:
            self._logger.warning("langfuse.client.init_failed", extra={"error": str(exc)})

    def _push_handle(self, handle: _LangfuseObservationHandle) -> None:
        trace_stack = list(self._current_trace.get() or [])
        span_stack = list(self._current_span.get() or [])
        if not trace_stack:
            trace_stack.append(handle)
        span_stack.append(handle)
        self._current_trace.set(trace_stack)
        self._current_span.set(span_stack)

    def _pop_handle(self, handle: _LangfuseObservationHandle) -> None:
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
        handle = self.start_observation(name=name)
        if handle is None:
            self._logger.info("trace.start.skipped", extra={"trace_name": name, "reason": "langfuse_disabled"})
            return None

        if propagate_attributes is not None:
            try:
                propagation_cm = propagate_attributes(
                    trace_name=trace_name,
                    user_id=user_id,
                    session_id=session_id,
                )
                propagation_cm.__enter__()
                handle.propagation_cm = propagation_cm
            except Exception as exc:
                self._logger.warning("trace.attributes.propagation_failed", extra={"error": str(exc)})

        self._logger.info(
            "trace.start",
            extra={
                "trace_name": trace_name or name,
                "trace_id": handle.trace_id,
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
        if self._client is None:
            return None

        starter = getattr(self._client, "start_as_current_observation", None)
        if not callable(starter):
            return None

        try:
            context_manager = starter(name=name, as_type=as_type, model=model)
        except TypeError as exc:
            # Le SDK Langfuse installé n'accepte pas un de ces kwargs
            # (ex: signature différente selon la version) -> on retente au minimum.
            self._logger.warning(
                "span.start.unsupported_kwargs",
                extra={"span_name": name, "error": str(exc)},
            )
            try:
                context_manager = starter(name=name)
            except Exception as exc2:
                self._logger.error("span.start.failed", extra={"span_name": name, "error": str(exc2)})
                return None
        except Exception as exc:
            self._logger.error("span.start.failed", extra={"span_name": name, "error": str(exc)})
            return None

        try:
            observation = context_manager.__enter__()
            # On récupère les trace et span id créés par le SDK Langfuse pour les stocker dans notre handle.
            trace_id = getattr(observation, "trace_id", None)
            span_id = getattr(observation, "id", None)
            handle = _LangfuseObservationHandle(
                adapter=self, # représente l'instance sur laquelle la méthode start_observation a été appelée.
                logger=self._logger,
                context_manager=context_manager, # pour fermeture du span
                observation=observation,
                trace_id=trace_id,
                observation_id=span_id,
                name=name,
            )
            self._push_handle(handle)
            return handle
        except Exception as exc:
            self._logger.error("span.start.failed", extra={"span_name": name, "error": str(exc)})
            return None

    def finish_trace(self) -> None:
        while self._current_span.get():
            handle = self._current_span.get()[-1]
            handle.close()
        while self._current_trace.get():
            handle = self._current_trace.get()[-1]
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
        if self._client is None:
            self._logger.info(
                "score.create.skipped",
                extra={"trace_id": trace_id, "observation_id": observation_id, "score_name": name, "value": value, "reason": "langfuse_disabled"},
            )
            return

        try:
            kwargs = {
                key: value
                for key, value in {
                    "trace_id": trace_id,
                    "name": name,
                    "value": value,
                    "comment": comment,
                    "observation_id": observation_id,
                }.items()
                if value is not None
            }

            self._client.create_score(**kwargs)
            if hasattr(self._client, "flush"):
                self._client.flush()
        except Exception as exc:
            self._logger.error(
                "score.create.failed",
                extra={"trace_id": trace_id, "observation_id": observation_id, "score_name": name, "value": value, "error": str(exc)},
            )

