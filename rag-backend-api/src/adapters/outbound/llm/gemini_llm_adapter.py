import os
from typing import List

from google import genai
from google.genai import types

from src.application.ports.outbound.llm_port import LlmPort
from src.application.ports.outbound.prompt_builder_port import PromptBuilderPort
from src.config.default_answer import DEFAULT_ANSWER
from src.config.prompt_gemini import SYSTEM_MESSAGE
from src.domain.model.rag_models import Document, GenerationResult


class GeminiLlmAdapter(LlmPort):
    def __init__(self, prompt_builder: PromptBuilderPort, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv("MODEL_NAME", "gemini-2.5-flash")
        self._prompt_builder = prompt_builder
        self._client = genai.Client()
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self._max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512"))
        self._thinking_budget = int(os.getenv("LLM_THINKING_BUDGET", "200"))
        self._seed = int(os.getenv("LLM_SEED", "42"))

    def generate_answer(self, question: str, chunks: List[Document]) -> GenerationResult:
        prompt = self._prompt_builder.build_prompt(question=question, chunks=chunks)
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_MESSAGE,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=self._thinking_budget),
                seed=self._seed,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )

        text = response.text if response.text else DEFAULT_ANSWER
        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
        thoughts_tokens = int(getattr(usage_metadata, "thoughts_token_count", 0) or 0)
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = getattr(candidates[0], "finish_reason", None)
            finish_reason = getattr(finish, "name", None)

        return GenerationResult(
            text=text,
            prompt=prompt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thoughts_tokens=thoughts_tokens,
            finish_reason=finish_reason,
        )

    def get_model_name(self) -> str | None:
        return self._model_name

    def get_generation_config(self) -> dict[str, int]:
        return {
            "thinking_budget": self._thinking_budget,
            "max_output_tokens": self._max_output_tokens,
        }
