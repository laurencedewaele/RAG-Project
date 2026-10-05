from typing import List

from src.application.ports.outbound.prompt_builder_port import PromptBuilderPort
from src.domain.model.rag_models import Document


class RagPromptBuilderAdapter(PromptBuilderPort):
    def build_prompt(self, question: str, chunks: List[Document]) -> str:
        context = "\n\n".join(chunk.content for chunk in chunks)
        return f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}"