from pydantic import BaseModel, Field


class SubmitFeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1)
    score_value: str = Field(..., min_length=1)
    feedback_type: str = Field(..., min_length=1)
    comment: str | None = None


class SubmitFeedbackResponse(BaseModel):
    status: str