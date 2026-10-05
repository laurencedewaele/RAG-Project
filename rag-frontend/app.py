import os
import socket
import uuid
from typing import List, Optional, Tuple

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
DEFAULT_USE_MMR = os.getenv("DEFAULT_USE_MMR", "true").lower() == "true"
DEFAULT_LAMBDA_MULT = float(os.getenv("DEFAULT_LAMBDA_MULT", "0.55"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.4"))
DEFAULT_TRACE_NAME = os.getenv("DEFAULT_TRACE_NAME", "hf-space-chat")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "anonymous")
DEFAULT_BUDGET = int(os.getenv("DEFAULT_BUDGET", "200"))
# Langfuse cloud instance URL (defaults to official cloud)
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_LINK = "https://cloud.langfuse.com/project/cmnvvu9nd00a1ad07pbjkmht0?filter=tags%3BarrayOptions%3B%3Bany+of%3Bmiss-terry"

custom_css = """
/* Hide fullscreen button */
#fixed_logo button[aria-label="Fullscreen"] {
    display: none !important;
}

/* Hide share button */
#fixed_logo button[aria-label="Share"] {
    display: none !important;
}
"""

def format_sources(chunks: List[dict]) -> str:
    if not chunks:
        return ""

    lines = ["### Sources récupérées :   \n"]
    for chunk in chunks:
        title = chunk.get("title") or "Source"
        score = chunk.get("score")
        header = f"{title} (score: {score:.3f})" if score is not None else title
        content = (chunk.get("content") or "").strip().replace("\n", "\n  ")
        lines.append(f"- **{header}**  \n  {content}")
    return "\n".join(lines)


def format_sources_html(chunks: List[dict]) -> str:
    if not chunks:
        return ""

    rows = []
    for chunk in chunks:
        title = chunk.get("title") or "Source"
        author = chunk.get("author") or "Inconnu"
        score = chunk.get("score")
        score_text = f"<span style='color:#4ea3ff;'>({score:.3f})</span>" if score is not None else ""
        content = (chunk.get("content") or "").strip().replace("\n", "<br>")
        rows.append(
            f"<tr><td style='padding:4px 0;'><strong>{title}</strong><br>{author}{score_text}</td></tr>"
        )

    return f"""
    <div style="max-height: 280px; overflow-y:auto; padding: 6px 0;">
        <table style="width:100%; border-collapse:collapse; color:#eee;">
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


def ask_backend(
    question: str,
    history: Optional[List[Tuple[str, str]]] = None,
    top_k: int = DEFAULT_TOP_K,
    use_mmr: bool = DEFAULT_USE_MMR,
    lambda_mult: float = DEFAULT_LAMBDA_MULT,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[List[Tuple[str, str]], Optional[str], List[dict]]:
    if not question or not question.strip():
        return (history or []), None, []

    payload = {
        "question": question.strip(),
        "top_k": top_k,
        "use_mmr": use_mmr,
        "lambda_mult": lambda_mult,
        "similarity_threshold": similarity_threshold,
        "trace_name": DEFAULT_TRACE_NAME,
        "user_id": DEFAULT_USER_ID,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/ask",
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        answer = f"Erreur de connexion au backend : {exc}"
        return (history or []) + [(question, answer)], None, []

    answer = data.get("answer", "Aucune réponse disponible.")
    chunks = data.get("chunks", [])

    trace_id = data.get("trace_id")
    return (history or []) + [(question, answer)], trace_id, chunks


def find_available_port(start_port: int, max_tries: int = 20) -> int:
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"Aucun port libre trouvé à partir de {start_port}")


def submit_feedback(
    trace_id: Optional[str],
    comment: str,
    feedback_type: str,
) -> Tuple[str, str]:
    if not trace_id:
        return "⚠️ Aucune réponse n’a encore été évaluée. Posez une question avant de donner votre avis.", comment

    payload = {
        "trace_id": trace_id,
        "score_value": "OK" if feedback_type == "helpful" else "KO",
        "feedback_type": feedback_type,
        "comment": comment.strip() if comment else None,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/feedback",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"❌ Échec de l’envoi du feedback : {exc}", comment

    return "✅ Feedback enregistré", ""


def start_conversation(prefix: str) -> str:
    return prefix + "_" + str(uuid.uuid4())


def init(profile: gr.OAuthProfile | None):
    if profile:
        return profile.username, f"**Connected User:** {profile.username}", \
               start_conversation(profile.username)
    return "anonymous", "**Connected User:** anonymous", "anonym_" + str(uuid.uuid4())


def on_clear(user: str):
    new_session = start_conversation(user)
    return ("", DEFAULT_TOP_K, [], "", [], "", True, DEFAULT_LAMBDA_MULT, None,
            [], "", "", new_session,)


def positive_feedback(trace_id: Optional[str], comment: str):
    return submit_feedback(trace_id, comment, "helpful")


def negative_feedback(trace_id: Optional[str], comment: str):
    return submit_feedback(trace_id, comment, "not_helpful")


def rag_pipeline(
    query: str,
    top_k: int,
    use_mmr: bool,
    lambda_slider: float,
    history: List,
    session_id: str,
    user_id: str,
    thinking_budget: int,
    similarity_threshold: float,
) -> Tuple[List[dict], str, List, List, Optional[str]]:

    updated_history, trace_id, chunks = ask_backend(
        question=query,
        history=history,
        top_k=top_k,
        use_mmr=use_mmr,
        lambda_mult=lambda_slider,
        similarity_threshold=similarity_threshold,
    )
    details_md = format_sources(chunks)
    return chunks, details_md, updated_history, updated_history, trace_id


with gr.Blocks(title="Chatbot Search", css=custom_css) as demo:
    #gr.LoginButton()

    user_state = gr.State("anonymous")
    session_state = gr.State(start_conversation("anonymous"))

    # Header with logo and title
    with gr.Row():
        with gr.Column():
            gr.Image(value="logo_got.webp", type="filepath",
                    width=50, interactive=False,
                    show_download_button=False, show_label=False)
        with gr.Column():
            gr.HTML("<h1 style='text-align: left; font-weight: bold;'>   Test et évaluation du chatbot</h1>")

    gr.Markdown(" ")

    with gr.Row():
        user_name = gr.Markdown("")
        gr.HTML(f"<a href='{LANGFUSE_LINK}' target='_blank'>Langfuse traces</a>")

    demo.load(init, inputs=None, outputs=[user_state, gr.Markdown(""), session_state])

    list_passages = gr.State()
    list_history = gr.State([])
    current_trace_id = gr.State()

    sel_query = gr.Textbox(label="Query")
    chatbot = gr.Chatbot()

    with gr.Row():
        feedback_comment = gr.Textbox(label="Comment (optional)", lines=1, scale=8)
        btn_like = gr.Button("👍 Helpful", scale=1)
        btn_dislike = gr.Button("👎 Not Helpful", scale=1)

    feedback_status = gr.Markdown()
    details_md = gr.Markdown()

    clear = gr.Button("Clear", variant="primary")

    with gr.Row():
        sel_top_k = gr.Slider(2, 20, value=DEFAULT_TOP_K, step=1, label="Top k", info="Number of passages to retrieve for semantic search.")
        with gr.Group():
            use_mmr = gr.Checkbox(label="Enable MMR", value=DEFAULT_USE_MMR, info="Enable Maximum Marginal Relevance (MMR) reranking")
            lambda_slider = gr.Slider(minimum=0, maximum=1, value=0.55, step=0.05, label="MMR Lambda", info="0 = more diversity (less redundancy), 1 = more relevance (closer matches)", interactive=False)
        use_mmr.change(fn=lambda active: gr.update(interactive=active), inputs=[use_mmr], outputs=[lambda_slider])

        with gr.Group():
            thinking_budget = gr.Slider(minimum=0, maximum=1024, value=DEFAULT_BUDGET, step=50, label="Thinking Budget (tokens)", info="Number of tokens allocated for the model's internal reasoning/thought process.")
        with gr.Group():
            similarity_threshold = gr.Slider(minimum=0, maximum=1, value=DEFAULT_SIMILARITY_THRESHOLD, step=0.05, label="Similarity Threshold", info="Minimum score required for a passage to be retrieved (1 = perfect match).")

    sel_query.submit(
        rag_pipeline,
        inputs=[sel_query, sel_top_k, use_mmr, lambda_slider, list_history, session_state, user_state, thinking_budget, similarity_threshold],
        outputs=[list_passages, details_md, list_history, chatbot, current_trace_id],
    ).then(lambda: "", outputs=[sel_query])

    btn_like.click(
        positive_feedback,
        inputs=[current_trace_id, feedback_comment],
        outputs=[feedback_status, feedback_comment],
    )

    btn_dislike.click(
        negative_feedback,
        inputs=[current_trace_id, feedback_comment],
        outputs=[feedback_status, feedback_comment],
    )

    clear.click(
        fn=on_clear,
        inputs=[user_state],
        outputs=[sel_query, sel_top_k, chatbot, details_md, chatbot, details_md, use_mmr, lambda_slider, list_passages, list_history, feedback_status, feedback_comment, session_state],
    )


if __name__ == "__main__":
    requested_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    server_port = find_available_port(requested_port)
    print(f"Démarrage du frontend sur http://127.0.0.1:{server_port}")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=server_port,
        debug=True,
        show_error=True,
        quiet=True,
    )
