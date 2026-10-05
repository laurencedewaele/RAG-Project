from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(slots=True)
class ScoreInput:
    name: str
    value: float | str
    data_type: str | None = None
    comment: str | None = None


class ObservationHandle(Protocol):
    trace_id: Optional[str]
    observation_id: Optional[str]

    def update(
        self,
        *,
        input: Any | None = None,
        output: Any | None = None,
        status_message: str | None = None,
        metadata: Dict[str, Any] | None = None,
        usage_details: Dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def score(self, scores: Dict[str, ScoreInput]) -> None:
        """Score multiple metrics at once.

        Args:
            scores: Dictionary where keys are score names and values are ScoreInput objects.
                   Example: {"latency": ScoreInput(name="latency", value=0.5, data_type="NUMERIC")}
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class ObservabilityPort(ABC):
    @abstractmethod
    def start_trace(
        self,
        name: str,
        payload: Dict[str, Any],
        *,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[ObservationHandle]:
        raise NotImplementedError

    @abstractmethod
    def start_observation(
        self,
        name: str,
        as_type: str | None = None,
        model: str | None = None,
    ) -> Optional[ObservationHandle]:
        raise NotImplementedError

    @abstractmethod
    def finish_trace(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_score(
        self,
        trace_id: str,
        name: str,
        value: float | str,
        comment: Optional[str] = None,
        observation_id: Optional[str] = None,
    ) -> None:
        raise NotImplementedError
