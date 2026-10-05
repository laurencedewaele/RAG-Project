from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UserFeedback:
    feedback_type: str
    feedback_comment: Optional[str] = None