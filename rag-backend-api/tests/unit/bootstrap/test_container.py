import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import src.bootstrap.container as container_module


class DummyLLM:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name

    def get_model_name(self) -> str | None:
        return self._model_name


class DummyEmbedding:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name

    def get_model_name(self) -> str | None:
        return self._model_name

    def get_embedding_function(self):
        return MagicMock()


def test_container_uses_specific_ragas_llm_and_embedding(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_MODEL_NAME", "ragas-llm")
    monkeypatch.setenv("RAGAS_EMBED_MODEL_NAME", "ragas-embed")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("ENABLE_RAGAS_EVAL", "true")

    created_ragas_kwargs = {}

    def fake_ragas_adapter(**kwargs):
        created_ragas_kwargs.update(kwargs)
        return SimpleNamespace()

    with patch.object(container_module, "HuggingFaceEmbeddingAdapter", side_effect=lambda model_name=None: DummyEmbedding(model_name=model_name)), \
         patch.object(container_module, "GeminiLlmAdapter", side_effect=lambda prompt_builder, model_name=None: DummyLLM(model_name=model_name)), \
         patch.object(container_module, "RagasEvaluationAdapter", side_effect=fake_ragas_adapter), \
         patch.object(container_module, "NoOpEvaluationAdapter", return_value=SimpleNamespace()), \
         patch.object(container_module, "NativeRetrievalAdapter", return_value=SimpleNamespace()), \
         patch.object(container_module, "RagPromptBuilderAdapter", return_value=SimpleNamespace()), \
         patch.object(container_module, "AskQuestionService", return_value=SimpleNamespace()), \
         patch.object(container_module, "SubmitFeedbackService", return_value=SimpleNamespace()), \
         patch.object(container_module, "LoggingObservabilityAdapter", return_value=SimpleNamespace()), \
         patch.object(container_module, "LangfuseObservabilityAdapter", return_value=SimpleNamespace()):
        container_module.Container()

    assert created_ragas_kwargs["llm_port"].get_model_name() == "ragas-llm"
    assert created_ragas_kwargs["embedding_port"].get_model_name() == "ragas-embed"
