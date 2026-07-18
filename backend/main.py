from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException, status

from backend.agents import AgentConfigurationError, run_advisor, run_enrichment
from backend.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    SessionResponse,
    SessionState,
    UserProfile,
)


class InMemorySessionStore:
    """Hackathon-only session storage; all state is lost when the API restarts."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, user_id: str) -> SessionState:
        async with self._lock:
            return self._sessions.setdefault(user_id, SessionState(user_id=user_id))

    async def get(self, user_id: str) -> SessionState | None:
        async with self._lock:
            return self._sessions.get(user_id)

    async def update(self, state: SessionState) -> None:
        async with self._lock:
            state.updated_at = datetime.utcnow()
            self._sessions[state.user_id] = state


app = FastAPI(title="Ask Insurance API", version="0.1.0")
sessions = InMemorySessionStore()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    state = await sessions.get_or_create(request.user_id)
    return SessionResponse(user_id=state.user_id, profile=state.profile, messages=state.messages)


@app.post("/sessions/{user_id}/messages", response_model=ChatResponse)
async def send_message(user_id: str, request: ChatRequest) -> ChatResponse:
    state = await sessions.get(user_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown user session.")

    user_message = request.message.strip()
    history = [{"role": item.role, "content": item.content} for item in state.messages]
    state.messages.append(ChatMessage(role="user", content=user_message))

    try:
        advisor_turn = await run_advisor(state.profile, history, user_message)
        if advisor_turn.scope_status == "out_of_scope":
            assistant_message = (
                "That’s outside my scope. I can help with choosing health insurance."
            )
            jargon, traps, recommendation = [], [], None
        else:
            _apply_profile_update(
                state.profile, advisor_turn.profile_updates.model_dump(exclude_none=True)
            )
            jargon_result, traps_result = await run_enrichment(
                state.profile,
                user_message,
                advisor_turn.assistant_message,
                advisor_turn.jargon_terms,
                advisor_turn.show_buying_checks,
            )
            assistant_message = advisor_turn.assistant_message
            jargon = jargon_result.explanations
            traps = traps_result.warnings
            recommendation = advisor_turn.recommendation
    except AgentConfigurationError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="The insurance assistant could not complete this reply. Please try again.",
        ) from error

    state.messages.append(
        ChatMessage(
            role="assistant",
            content=assistant_message,
            jargon=jargon,
            warnings=traps,
            recommendation=recommendation,
        )
    )
    await sessions.update(state)
    return ChatResponse(
        user_id=user_id,
        assistant_message=assistant_message,
        profile=state.profile,
        jargon=jargon,
        warnings=traps,
        recommendation=recommendation,
    )


def _apply_profile_update(profile: UserProfile, updates: dict[str, object]) -> None:
    for field_name, value in updates.items():
        if value is not None:
            setattr(profile, field_name, value)
