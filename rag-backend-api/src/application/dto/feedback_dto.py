from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SubmitFeedbackCommand:
    trace_id: str
    score_value: str
    feedback_type: str
    comment: Optional[str] = None


@dataclass(frozen=True)
class SubmitFeedbackResult:
    status: str