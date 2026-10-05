import logging
import os
import sys
from queue import Queue
from threading import Thread
from typing import Any, Optional
from unittest.mock import MagicMock

# Avoid import failures in some environments where Ragas probes Vertex AI modules.
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())
sys.modules.setdefault("langchain_community.llms.vertexai", MagicMock())

from openai import AsyncOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, Faithfulness

from src.application.ports.outbound.evaluation_port import EvaluationPort
from src.application.ports.outbound.observability_port import ObservabilityPort
from src.domain.model.evaluation_models import AnswerEvaluation, EvaluationResult


class PendingEvaluation:
    def __init__(
        self,
        evaluation: AnswerEvaluation,
        trace_id: Optional[str],
        observation_id: Optional[str] = None,
    ) -> None:
        self.evaluation = evaluation
        self.trace_id = trace_id
        self.observation_id = observation_id


class RagasEvaluationAdapter(EvaluationPort):
    def __init__(
        self,
        observability_port: ObservabilityPort,
        llm_name: str,
        embedding_model_name: str,
    ) -> None:
        self._logger = logging.getLogger("chatbot.backend.ragas")
        self._obs = observability_port
        self._queue: Queue[PendingEvaluation] = Queue()

        cache = DiskCacheBackend()
        # OpenAI models require passing an AsyncOpenAI client to ragas llm_factory.
        if llm_name.startswith(("gpt", "o1", "o3", "o4")):
            self._evaluator_llm = llm_factory(llm_name, cache=cache, client=AsyncOpenAI())
        else:
            self._evaluator_llm = llm_factory(llm_name, cache=cache)

        ragas_embeddings = self._build_ragas_embeddings(model_name=embedding_model_name)
        self._metrics = [
            Faithfulness(llm=self._evaluator_llm),
            AnswerRelevancy(llm=self._evaluator_llm, embeddings=ragas_embeddings)
            if ragas_embeddings is not None
            else AnswerRelevancy(llm=self._evaluator_llm),
        ]

        self._worker = Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _build_ragas_embeddings(self, model_name: str) -> Any | None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper

            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            return LangchainEmbeddingsWrapper(embeddings)
        except Exception as exc:
            self._logger.warning(
                "ragas.evaluation.embeddings_init_failed",
                extra={"error": str(exc), "embedding_model": model_name},
            )
            return None

    def enqueue(
        self,
        evaluation: AnswerEvaluation,
        trace_id: Optional[str],
        observation_id: Optional[str] = None,
    ) -> None:
        if not evaluation.contexts:
            return

        self._queue.put(
            PendingEvaluation(evaluation=evaluation, trace_id=trace_id, observation_id=observation_id)
        )

    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                self._process_item(item)
            except Exception as exc:
                self._logger.error("ragas.evaluation.failed", extra={"error": str(exc)})
            finally:
                self._queue.task_done()

    def _process_item(self, pending: PendingEvaluation) -> None:
        evaluation = pending.evaluation
        dataset = Dataset.from_dict(
            {
                "question": [evaluation.question],
                "answer": [evaluation.answer],
                "contexts": [evaluation.contexts],
            }
        )

        result = evaluate(dataset=dataset, metrics=self._metrics)
        raw_scores = result.scores[0]
        scores: dict[str, float] = {}
        for score_name, raw_value in raw_scores.items():
            if raw_value is None:
                value = 0.0
            else:
                parsed = float(raw_value)
                value = 0.0 if parsed != parsed else parsed
            scores[score_name] = value

        evaluation_result = EvaluationResult(scores=scores)

        trace_id = pending.trace_id
        if not trace_id:
            self._logger.info("ragas.evaluation.skip_scoring", extra={"reason": "missing_trace_id"})
            return

        for score_name, value in evaluation_result.scores.items():
            self._obs.create_score(
                trace_id=trace_id,
                name=score_name,
                value=value,
                observation_id=pending.observation_id,
            )

        self._logger.info(
            "ragas.evaluation.success",
            extra={"trace_id": trace_id, "scores": evaluation_result.scores},
        )
