import time
import os
import uuid

from src.application.dto.ask_question_dto import AskQuestionCommand, AskQuestionResult, ChunkView
from src.application.ports.outbound.evaluation_port import EvaluationPort
from src.application.ports.inbound.ask_question_use_case import AskQuestionUseCase
from src.application.ports.outbound.llm_port import LlmPort
from src.application.ports.outbound.observability_port import ObservabilityPort, ScoreInput
from src.application.ports.outbound.retrieval_port import RetrievalPort
from src.domain.errors.exceptions import EmptyQuestionError
from src.domain.model.evaluation_models import AnswerEvaluation
from src.domain.model.rag_models import GenerationResult


MODEL_PRICING_PER_1K = {
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "Qwen/Qwen2.5-3B-Instruct": {"input": 0.0, "output": 0.0},
}


class AskQuestionService(AskQuestionUseCase):
    def __init__(
        self,
        retrieval_port: RetrievalPort,
        llm_port: LlmPort,
        observability_port: ObservabilityPort,
        default_answer: str,
        evaluation_port: EvaluationPort | None = None,
    ) -> None:
        self._retrieval = retrieval_port
        self._llm = llm_port
        self._obs = observability_port
        self._default_answer = default_answer
        self._evaluation = evaluation_port

    def execute(self, command: AskQuestionCommand) -> AskQuestionResult:
        if not command.question or not command.question.strip():
            raise EmptyQuestionError("Question is empty")

        lambda_mult = command.lambda_mult if command.use_mmr else None
        start_total = time.perf_counter()
        model_name = self._llm.get_model_name()
        pricing = MODEL_PRICING_PER_1K.get(model_name or "", {"input": 0.0, "output": 0.0})

        # A new session_id is generated for every API call: the backend is stateless,
        # unlike app.py which keeps one session_id per Gradio conversation.
        session_id = str(uuid.uuid4())

        trace_handle = self._obs.start_trace(
            name="rag-pipeline",
            payload={
                "query": command.question,
                "top_k": command.top_k,
                "use_mmr": command.use_mmr,
                "lambda_mult": lambda_mult,
                "similarity_threshold": command.similarity_threshold,
            },
            trace_name=command.trace_name,
            user_id=command.user_id,
            session_id=session_id,
        )

        retrieval_handle = self._obs.start_observation(name="retrieval", as_type="retriever")
        generation_handle = None
        try:
            retrieval_start = time.perf_counter()
            list_chunks = self._retrieval.retrieve(
                question=command.question,
                top_k=command.top_k,
                lambda_mult=lambda_mult,
                similarity_threshold=command.similarity_threshold,
            )
            retrieval_latency = time.perf_counter() - retrieval_start
            max_len_chunk = max((len(chunk.content) for chunk in list_chunks), default=0)
            total_doc_length = sum(len(chunk.content) for chunk in list_chunks)

            if retrieval_handle is not None:
                retrieval_handle.update(
                    input={"query": command.question},
                    output={"retrieved_chunks": [chunk.content for chunk in list_chunks]},
                    status_message="SUCCESS",
                    metadata={
                        "reranker_used": False,
                        "mmr_used": command.use_mmr,
                        "top_k": str(command.top_k),
                        "mmr_lambda": str(lambda_mult) if command.use_mmr else None,
                        "reranker_name": None,
                        "similarity_threshold": str(command.similarity_threshold) if command.similarity_threshold is not None else None,
                        "retrieval_api_url": os.getenv("API_URL"),
                    },
                )
                retrieval_scores = {
                    "retrieve_latency": ScoreInput(name="retrieve_latency", value=round(retrieval_latency, 2)),
                    "max_len_chunk": ScoreInput(name="max_len_chunk", value=int(max_len_chunk)),
                    "total_doc_length": ScoreInput(name="total_doc_length", value=int(total_doc_length)),
                }
                retrieval_handle.score(retrieval_scores)
                retrieval_handle.close()
                retrieval_handle = None # For the finally block
        except Exception as exc:
            if retrieval_handle is not None:
                retrieval_handle.update(input={"query": command.question}, output=str(exc), status_message="ERROR")
            if trace_handle is not None:
                trace_handle.update(input={"query": command.question}, output=str(exc), status_message="ERROR")
            raise

        try:
            generation_handle = self._obs.start_observation(name="answer-generation", as_type="generation", model=model_name)

            if not list_chunks:
                answer = self._default_answer
                generation_result = GenerationResult(
                    text=answer,
                    prompt="",
                    input_tokens=0,
                    output_tokens=0,
                    thoughts_tokens=0,
                    finish_reason=None,
                )
                if generation_handle is not None:
                    generation_handle.update(
                        input={"query": command.question},
                        output={"answer": answer},
                        status_message="DEFAULT_ANSWER",
                    )
                if trace_handle is not None:
                    trace_handle.update(
                        input={"query": command.question},
                        output={"answer": answer},
                        status_message="DEFAULT_ANSWER")
            else:
                generation_start = time.perf_counter()
                generation_result = self._llm.generate_answer(question=command.question, chunks=list_chunks)
                generation_latency = time.perf_counter() - generation_start
                answer = generation_result.text
                if generation_handle is not None:
                    generation_config = self._llm.get_generation_config()
                    generation_handle.update(
                        input={"prompt": generation_result.prompt},
                        output={"answer": answer},
                        status_message="SUCCESS",
                        metadata={
                            "finish_reason": generation_result.finish_reason,
                            "prompt_version_file": "prompt_rag",
                            "thinking_budget": generation_config.get("thinking_budget"),
                            "max_output_tokens": generation_config.get("max_output_tokens"),
                        },
                        usage_details={
                            "input_tokens": int(generation_result.input_tokens),
                            "output_tokens": int(generation_result.output_tokens),
                            "thoughts_tokens": int(generation_result.thoughts_tokens),
                        },
                    )
                    if trace_handle is not None:
                        trace_handle.update(
                            input={"query": command.question},
                            output={"answer": answer},
                            status_message="SUCCESS",
                        )

            output_tokens_total = generation_result.output_tokens + generation_result.thoughts_tokens
            tokens_per_sec = int(output_tokens_total / generation_latency) if generation_latency > 0 else 0
            input_cost = (generation_result.input_tokens / 1000.0) * pricing["input"]
            output_cost = (output_tokens_total / 1000.0) * pricing["output"]

            if generation_handle is not None:
                generation_scores = {
                    "model_latency": ScoreInput(name="model_latency", value=round(generation_latency, 2), data_type="NUMERIC"),
                    "tokens_per_sec": ScoreInput(name="tokens_per_sec", value=int(tokens_per_sec), data_type="NUMERIC"),
                    "input_tokens": ScoreInput(name="input_tokens", value=int(generation_result.input_tokens), data_type="NUMERIC"),
                    "output_tokens": ScoreInput(name="output_tokens", value=int(output_tokens_total), data_type="NUMERIC"),
                    "total_cost": ScoreInput(name="total_cost", value=float(input_cost + output_cost), data_type="NUMERIC"),
                }
                generation_handle.score(generation_scores)
        except Exception as exc:
            if generation_handle is not None:
                generation_handle.update(input={"query": command.question}, output=str(exc), status_message="ERROR")
            if trace_handle is not None:
                trace_handle.update(input={"query": command.question}, output=str(exc), status_message="ERROR")
            raise

        try:
            total_latency = time.perf_counter() - start_total

            if trace_handle is not None:
                trace_handle.score({"total_latency": ScoreInput(name="total_latency", value=float(round(total_latency, 2)))})

            if self._evaluation is not None:
                self._evaluation.enqueue(
                    AnswerEvaluation(
                        question=command.question,
                        answer=answer,
                        contexts=[chunk.content for chunk in list_chunks],
                    ),
                    trace_id=trace_handle.trace_id if trace_handle else None,
                    observation_id=getattr(generation_handle, "observation_id", None),
                )

            return AskQuestionResult(
                answer=answer,
                chunks=[
                    ChunkView(
                        id=chunk.id,
                        content=chunk.content,
                        score=chunk.score,
                        title=chunk.title,
                        author=chunk.author,
                    )
                    for chunk in list_chunks
                ],
                trace_id=trace_handle.trace_id if trace_handle else None,
            )
        except Exception as exc:
            if trace_handle is not None:
                trace_handle.update(input={"query": command.question}, output=str(exc), status_message="ERROR")
            raise
        finally:
            if generation_handle is not None:
                generation_handle.close()
            if retrieval_handle is not None:
                retrieval_handle.close()
            if trace_handle is not None:
                trace_handle.close()
            self._obs.finish_trace()
