from pydantic import BaseModel, Field


class AskQuestionRequest(BaseModel):
    question: str = Field(default="qui est miss terry ?", min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    use_mmr: bool = True
    lambda_mult: float | None = Field(default=0.55, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    trace_name: str = Field(default="chatbot-search", min_length=1)
    user_id: str = Field(default="anonymous", min_length=1)


class ChunkResponse(BaseModel):
    id: str
    content: str
    score: float | None
    title: str | None
    author: str | None


class AskQuestionResponse(BaseModel):
    answer: str
    chunks: list[ChunkResponse]
    trace_id: str | None
