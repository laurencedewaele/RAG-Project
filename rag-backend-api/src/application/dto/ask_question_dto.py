from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class AskQuestionCommand:
    question: str
    top_k: int = 3
    use_mmr: bool = False
    lambda_mult: Optional[float] = None
    similarity_threshold: float = 0.4
    trace_name: str = "chatbot-search"
    user_id: str = "anonymous"


@dataclass(frozen=True)
class ChunkView:
    id: str
    content: str
    score: Optional[float]
    title: Optional[str]
    author: Optional[str]


@dataclass(frozen=True)
class AskQuestionResult:
    answer: str
    chunks: List[ChunkView]
    trace_id: Optional[str]
