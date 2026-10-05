from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Document:
    id: str
    content: str
    score: Optional[float]
    title: Optional[str] = None
    author: Optional[str] = None


@dataclass(frozen=True)
class Answer:
    text: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt: str
    input_tokens: int
    output_tokens: int
    thoughts_tokens: int
    finish_reason: Optional[str] = None
