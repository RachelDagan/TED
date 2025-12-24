import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openai import OpenAI
from pinecone import Pinecone
from fastapi.responses import RedirectResponse

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Assignment constraints
    chunk_size_tokens: int = int(os.environ.get("CHUNK_SIZE", "768"))   # <= 2048
    overlap_ratio: float = float(os.environ.get("OVERLAP_RATIO", "0.2")) # 0..0.3
    top_k: int = int(os.environ.get("TOP_K", "5"))                      # 1..30

    # Pinecone
    index_name: str = os.environ.get("PINECONE_INDEX_NAME", "ted-talks-index")
    namespace: str = os.environ.get("PINECONE_NAMESPACE", "ted")

    # LLMod / Models
    llmod_base_url: str = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small")
    chat_model: str = os.environ.get("CHAT_MODEL", "RPRTHPB-gpt-5-mini")

    # Keys
    llm_key: Optional[str] = (os.environ.get("LLMOD_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    pinecone_key: Optional[str] = os.environ.get("PINECONE_API_KEY")


S = Settings()

# -----------------------
# REQUIRED SYSTEM PROMPT SECTION
# -----------------------
SYSTEM_PROMPT_REQUIRED_SECTION = (
    "You are a TED Talk assistant that answers questions strictly and\n"
    "only based on the TED dataset context provided to you (metadata\n"
    "and transcript passages). You must not use any external\n"
    "knowledge, the open internet, or information that is not explicitly\n"
    "contained in the retrieved context. If the answer cannot be\n"
    "determined from the provided context, respond: \"I don't know\n"
    "based on the provided TED data.\" Always explain your answer\n"
    "using the given context, quoting or paraphrasing the relevant\n"
    "transcript or metadata when helpful."
)

SYSTEM_PROMPT = (
    SYSTEM_PROMPT_REQUIRED_SECTION
    + "\n\n"
    "Formatting requirements:\n"
    "- If the user asks for a list of exactly 3 talk titles, return exactly 3 distinct titles (different talk_id).\n"
    "- When possible, cite excerpts as [1], [2], [3] corresponding to the provided context items.\n"
    "- Be concise."
)

app = FastAPI(title="TED RAG API (LLMod + Pinecone)")

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/")
def root():
    return {"message": "API is running"}

class PromptPayload(BaseModel):
    question: str


class Runtime:
    client: Optional[OpenAI] = None
    pinecone: Optional[Pinecone] = None
    index: Any = None


R = Runtime()


def _check_constraints() -> None:
    if S.chunk_size_tokens > 2048:
        raise RuntimeError("CHUNK_SIZE exceeds 2048 (assignment constraint).")
    if not (0.0 <= S.overlap_ratio <= 0.3):
        raise RuntimeError("OVERLAP_RATIO must be between 0 and 0.3 (assignment constraint).")
    if not (1 <= S.top_k <= 30):
        raise RuntimeError("TOP_K must be between 1 and 30 (assignment constraint).")


def _init_runtime() -> None:
    if not S.llm_key:
        raise RuntimeError("Missing LLM API key (LLMOD_API_KEY or OPENAI_API_KEY).")
    if not S.pinecone_key:
        raise RuntimeError("Missing PINECONE_API_KEY.")

    R.client = OpenAI(api_key=S.llm_key, base_url=S.llmod_base_url)
    R.pinecone = Pinecone(api_key=S.pinecone_key)
    R.index = R.pinecone.Index(S.index_name)


def _llm() -> OpenAI:
    if R.client is None:
        raise RuntimeError("LLM client not initialized.")
    return R.client


def _pc_index():
    if R.index is None:
        raise RuntimeError("Pinecone index not initialized.")
    return R.index


def _need_exact_three_titles(q: str) -> bool:
    ql = (q or "").lower()
    return ("exactly 3" in ql and "title" in ql) or ("return a list of exactly 3" in ql)


def _embed_text(text: str) -> List[float]:
    res = _llm().embeddings.create(model=S.embedding_model, input=text)
    return res.data[0].embedding


def _query_distinct_by_talk_id(question: str, want: int) -> List[Dict[str, Any]]:
    """
    Query Pinecone and return up to `want` items, each with a distinct talk_id.
    """
    # pull extra so we can deduplicate
    raw_k = min(30, max(S.top_k, 3) * 8)

    res = _pc_index().query(
        vector=_embed_text(question),
        top_k=raw_k,
        include_metadata=True,
        namespace=S.namespace,
    )

    # pick best chunk per talk_id
    best: Dict[str, Dict[str, Any]] = {}

    for m in (res.matches or []):
        md = m.metadata or {}
        talk_id = str(md.get("talk_id", "")).strip()
        if not talk_id:
            continue

        candidate = {
            "talk_id": talk_id,
            "title": str(md.get("title", "")).strip(),
            "chunk": str(md.get("chunk", "")).strip(),
            "score": float(m.score) if m.score is not None else 0.0,
        }

        if talk_id not in best or candidate["score"] > best[talk_id]["score"]:
            best[talk_id] = candidate

    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:want]


def _format_context_for_user_prompt(ctx: List[Dict[str, Any]], question: str) -> str:
    blocks = ["Use ONLY the following TED talk excerpts to answer the question.\n"]
    for i, item in enumerate(ctx, start=1):
        blocks.append(
            f"[{i}] Title: {item.get('title','')} | talk_id: {item.get('talk_id','')}\n"
            f"Excerpt: {item.get('chunk','')}\n"
        )
    blocks.append(f"Question: {question}")
    return "\n".join(blocks)


def _answer_with_rag(question: str, ctx: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_prompt = _format_context_for_user_prompt(ctx, question)

    chat = _llm().chat.completions.create(
        model=S.chat_model,
        temperature=1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "response": (chat.choices[0].message.content or "").strip(),
        "context": ctx,
        "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
    }


# -----------------------
# Startup
# -----------------------
@app.on_event("startup")
def on_startup():
    _check_constraints()
    _init_runtime()


# -----------------------
# Endpoints
# -----------------------
@app.get("/api/stats")
def api_stats():
    return {
        "chunk_size": S.chunk_size_tokens,
        "overlap_ratio": S.overlap_ratio,
        "top_k": S.top_k,
    }


@app.post("/api/prompt")
def api_prompt(payload: PromptPayload) -> Dict[str, Any]:
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")

    try:
        want = 3 if _need_exact_three_titles(question) else S.top_k
        ctx = _query_distinct_by_talk_id(question, want=want)
        return _answer_with_rag(question, ctx)

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.get("/health")
def health():
    ok = (R.client is not None) and (R.pinecone is not None) and (R.index is not None)

    stats = None
    try:
        s = _pc_index().describe_index_stats()
        # convert Pinecone response to a plain JSON-serializable dict
        if hasattr(s, "to_dict"):
            stats = s.to_dict()
        elif hasattr(s, "model_dump"):
            stats = s.model_dump()
        else:
            stats = s  # fallback
    except Exception:
        stats = None

    return {
        "ok": ok,
        "index_name": S.index_name,
        "namespace": S.namespace,
        "models": {"embedding": S.embedding_model, "chat": S.chat_model},
        "pinecone_stats": stats,
    }
