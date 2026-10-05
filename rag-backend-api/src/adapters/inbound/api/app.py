import logging
import time
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.application.dto.ask_question_dto import AskQuestionCommand
from src.application.dto.feedback_dto import SubmitFeedbackCommand
from src.domain.errors.exceptions import EmptyQuestionError
from src.adapters.inbound.api.feedback_schemas import SubmitFeedbackRequest, SubmitFeedbackResponse
from src.adapters.inbound.api.schemas import AskQuestionRequest, AskQuestionResponse, ChunkResponse
from src.bootstrap.container import Container

def create_app(container: Container | None = None) -> FastAPI:
    app = FastAPI(title="RAG Backend API", version="0.1.0")
    runtime_container = container or Container()
    logger = logging.getLogger("chatbot.backend.inbound")

    def get_container() -> Container:
        return runtime_container

    @app.get("/")
    def welcome() -> dict[str, str]:
        return {
            "message": "Bienvenue sur l'API RAG.",
            "docs": "/docs",
            "health": "/health",
        }

    @app.middleware("http")
    async def inbound_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "http.request.failure",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise

        def log_request(status_code: int) -> None:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "http.request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )

        response.headers["x-request-id"] = request_id

        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            log_request(response.status_code)
            return response

        async def wrapped_body_iterator():
            try:
                async for chunk in body_iterator:
                    yield chunk
            finally:
                log_request(response.status_code)

        response.body_iterator = wrapped_body_iterator()
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/ask", response_model=AskQuestionResponse)
    async def ask_question(
        payload: AskQuestionRequest,
        container: Container = Depends(get_container),
    ) -> AskQuestionResponse:
        try:
            command = AskQuestionCommand(
                question=payload.question,
                top_k=payload.top_k,
                use_mmr=payload.use_mmr,
                lambda_mult=payload.lambda_mult,
                similarity_threshold=payload.similarity_threshold,
                trace_name=payload.trace_name,
                user_id=payload.user_id,
            )
            result = await run_in_threadpool(
                container.ask_question.execute,
                command,
            )
            return AskQuestionResponse(
                answer=result.answer,
                trace_id=result.trace_id,
                chunks=[
                    ChunkResponse(
                        id=chunk.id,
                        content=chunk.content,
                        score=chunk.score,
                        title=chunk.title,
                        author=chunk.author,
                    )
                    for chunk in result.chunks
                ],
            )
        except EmptyQuestionError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        except Exception as exc:
            logger.exception("ask_question_failure")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @app.post("/api/v1/feedback", response_model=SubmitFeedbackResponse)
    async def submit_feedback(
        payload: SubmitFeedbackRequest,
        container: Container = Depends(get_container),
    ) -> SubmitFeedbackResponse:
        try:
            command = SubmitFeedbackCommand(
                trace_id=payload.trace_id,
                score_value=payload.score_value,
                feedback_type=payload.feedback_type,
                comment=payload.comment,
            )
            result = await run_in_threadpool(
                container.submit_feedback.execute,
                command,
            )
            return SubmitFeedbackResponse(status=result.status)
        except Exception as exc:
            logger.exception("submit_feedback_failure")
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return app


app = create_app()
