from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pymongo import ASCENDING

# Setup backend logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ask-insurance-api")

from backend.agents import (
    AgentConfigurationError,
    run_advisor,
    run_enrichment,
    run_policy_comparison,
    run_policy_insight,
)
from backend.catalog import (
    catalog_status,
    ensure_database_setup,
    get_database,
    get_policy_context,
    ingest_products,
    semantic_search,
)
from backend.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CatalogProduct,
    CompareProductsRequest,
    CreateSessionRequest,
    IngestCatalogRequest,
    IngestionResponse,
    PolicyComparison,
    PolicyInsight,
    SessionResponse,
    SessionState,
    SuggestedProduct,
    UserProfile,
)

CATALOG_SEED_PATH = Path(__file__).resolve().parents[1] / "data" / "product_catalog.json"


class MongoSessionStore:
    """MongoDB-backed session store to persist user sessions and chat state across API restarts."""

    def __init__(self) -> None:
        self.collection_name = "user_sessions"

    def _get_collection(self):
        # We fetch the DB instance on-demand through get_database()
        return get_database()[self.collection_name]

    async def get_or_create(self, user_id: str) -> SessionState:
        loop = asyncio.get_running_loop()
        collection = self._get_collection()
        
        def run() -> SessionState:
            doc = collection.find_one({"user_id": user_id})
            if doc is None:
                state = SessionState(user_id=user_id)
                collection.replace_one(
                    {"user_id": user_id},
                    state.model_dump(),
                    upsert=True
                )
                return state
            return SessionState.model_validate(doc)
            
        return await loop.run_in_executor(None, run)

    async def get(self, user_id: str) -> SessionState | None:
        loop = asyncio.get_running_loop()
        collection = self._get_collection()
        
        def run() -> SessionState | None:
            doc = collection.find_one({"user_id": user_id})
            if doc is None:
                return None
            return SessionState.model_validate(doc)
            
        return await loop.run_in_executor(None, run)

    async def update(self, state: SessionState) -> None:
        loop = asyncio.get_running_loop()
        collection = self._get_collection()
        state.updated_at = datetime.now(UTC)
        
        def run() -> None:
            collection.replace_one(
                {"user_id": state.user_id},
                state.model_dump(),
                upsert=True
            )
            
        await loop.run_in_executor(None, run)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up Mongo collections without preventing chat startup during a DB outage."""
    logger.info("Initializing application lifespan setup...")
    app.state.catalog_setup_error = None
    app.state.catalog_setup = None
    try:
        app.state.catalog_setup = await asyncio.to_thread(ensure_database_setup)
        # Ensure index for session store is created
        database = get_database()
        database["user_sessions"].create_index([("user_id", ASCENDING)], unique=True)
        logger.info(f"Database setup complete. Status details: {app.state.catalog_setup}")
    except Exception as error:
        app.state.catalog_setup_error = str(error)
        logger.error(f"Lifespan startup database setup failed: {error}", exc_info=True)
    yield
    logger.info("Shutting down application lifespan...")


app = FastAPI(title="Ask Insurance API", version="0.2.0", lifespan=lifespan)
sessions = MongoSessionStore()


@app.get("/")
async def root() -> dict[str, str]:
    """Service landing endpoint for platform health checks."""
    return {"status": "ok", "service": "ask-insurance-api"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _require_ingestion_key(x_admin_key: str | None = Header(default=None)) -> None:
    """Protect catalog writes when INGEST_API_KEY is configured in production."""
    expected_key = os.getenv("INGEST_API_KEY")
    if expected_key and x_admin_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid ingestion API key.")


@app.get("/catalog/status")
async def get_catalog_status() -> dict[str, object]:
    try:
        details = await asyncio.to_thread(catalog_status)
        details["startup_setup_error"] = app.state.catalog_setup_error
        details["startup_setup"] = app.state.catalog_setup
        return details
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error


@app.get("/catalog/search", response_model=list[SuggestedProduct])
async def search_catalog(
    query: str = Query(min_length=3, max_length=4_000),
    limit: int = Query(default=3, ge=1, le=10),
) -> list[SuggestedProduct]:
    try:
        return await asyncio.to_thread(semantic_search, query, limit=limit)
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error


@app.get("/catalog/products/{product_id}/insight", response_model=PolicyInsight)
async def product_insight(product_id: str) -> PolicyInsight:
    """Summarize the stored policy wording excerpts for one selected product."""
    try:
        context = await asyncio.to_thread(get_policy_context, product_id)
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    if context is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown product.")
    try:
        insight = await run_policy_insight(context)
        # The product identity and source state come from MongoDB, never from model output.
        insight.product_id = product_id
        insight.insurance_company = str(context["insurance_company"])
        insight.product = str(context["product"])
        insight.source_status = str(context["source_status"])
        insight.pdf_source = str(context["pdf_source"])
        return insight
    except AgentConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not analyze this policy wording.") from error


@app.post("/catalog/compare", response_model=PolicyComparison)
async def compare_products(request: CompareProductsRequest) -> PolicyComparison:
    """Compare selected products against their stored policy-wording excerpts."""
    if len(set(request.product_ids)) != len(request.product_ids):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Select distinct products.")
    try:
        contexts = await asyncio.gather(
            *(asyncio.to_thread(get_policy_context, product_id) for product_id in request.product_ids)
        )
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    if any(context is None for context in contexts):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more selected products no longer exist.")
    policy_contexts = [context for context in contexts if context is not None]
    try:
        comparison = await run_policy_comparison(policy_contexts)
        comparison.product_ids = request.product_ids
        comparison.product_labels = [
            f"{context['insurance_company']} — {context['product']}" for context in policy_contexts
        ]
        for row in comparison.rows:
            if len(row.values) != len(policy_contexts):
                row.values = ["Not found in the supplied excerpts"] * len(policy_contexts)
        return comparison
    except AgentConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not compare the selected policy wordings.") from error


@app.post("/catalog/ingest", response_model=IngestionResponse, dependencies=[Depends(_require_ingestion_key)])
async def ingest_catalog(request: IngestCatalogRequest) -> IngestionResponse:
    """Upsert product metadata, extract its public wording, embed chunks, and index them."""
    try:
        results = await asyncio.to_thread(
            ingest_products,
            request.products,
            fetch_policy_wording=request.fetch_policy_wordings,
        )
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    return IngestionResponse(
        results=results,
        indexed=sum(result.status == "indexed" for result in results),
        metadata_only=sum(result.status == "metadata_only" for result in results),
        failed=sum(result.status == "failed" for result in results),
    )


@app.post("/catalog/seed", response_model=IngestionResponse, dependencies=[Depends(_require_ingestion_key)])
async def seed_supplied_catalog(fetch_policy_wordings: bool = Query(default=True)) -> IngestionResponse:
    """Ingest the supplied insurer catalog committed at data/product_catalog.json."""
    try:
        products = [CatalogProduct.model_validate(item) for item in json.loads(CATALOG_SEED_PATH.read_text())]
        results = await asyncio.to_thread(
            ingest_products, products, fetch_policy_wording=fetch_policy_wordings
        )
    except Exception as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    return IngestionResponse(
        results=results,
        indexed=sum(result.status == "indexed" for result in results),
        metadata_only=sum(result.status == "metadata_only" for result in results),
        failed=sum(result.status == "failed" for result in results),
    )


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    logger.info(f"Creating session for user_id: {request.user_id}")
    state = await sessions.get_or_create(request.user_id)
    return SessionResponse(user_id=state.user_id, profile=state.profile, messages=state.messages)


@app.post("/sessions/{user_id}/messages", response_model=ChatResponse)
async def send_message(user_id: str, request: ChatRequest) -> ChatResponse:
    logger.info(f"Received message for user_id {user_id}. Content length: {len(request.message)}")
    state = await sessions.get(user_id)
    if state is None:
        logger.warning(f"Session not found for user_id: {user_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown user session.")

    user_message = request.message.strip()
    history = [{"role": item.role, "content": item.content} for item in state.messages]
    state.messages.append(ChatMessage(role="user", content=user_message))

    try:
        # Check if we should pass previous suggested products into the conversation context
        # (e.g. if the user is asking a follow-up question comparing the products)
        previous_suggested = []
        for msg in reversed(state.messages):
            if msg.role == "assistant" and msg.suggested_products:
                previous_suggested = [
                    {
                        "insurance_company": p.insurance_company,
                        "product": p.product,
                        "primary_use_case": p.primary_use_case,
                        "pdf_source": str(p.pdf_source),
                        "relevance_score": p.relevance_score,
                    }
                    for p in msg.suggested_products
                ]
                break

        # Check if any products were selected by the user for comparison previously,
        # so we can append that selection context for the LLM
        # We find user selections by parsing conversation text or state clues if present.
        # This keeps the selection history directly available within the prompt's candidate list.
        if previous_suggested:
            # We can label selected products directly in the candidate metadata we send to Gemini
            for p in previous_suggested:
                p["is_selected_by_user_for_comparison"] = True

        logger.info(f"Invoking Gemini Advisor Agent for user_id {user_id}...")
        advisor_turn = await run_advisor(state.profile, history, user_message, previous_suggested)
        logger.info(f"Advisor response generated. Scope status: {advisor_turn.scope_status}")
        
        if advisor_turn.scope_status == "out_of_scope":
            assistant_message = (
                "That’s outside my scope. I can help with choosing health insurance."
            )
            jargon, traps, recommendation = [], [], None
            suggested_products = []
            should_suggest_products = False
        else:
            _apply_profile_update(
                state.profile, advisor_turn.profile_updates.model_dump(exclude_none=True)
            )
            is_final_recommendation = (
                advisor_turn.ready_for_recommendation
                and advisor_turn.recommendation is not None
            )
            # Determine whether we should display suggested products.
            # We trust the advisor_turn's should_suggest_products flag or final recommendation status.
            should_suggest_products = (
                advisor_turn.should_suggest_products or is_final_recommendation
            )

            # Keep suggestions highly dynamic:
            # If the LLM has already generated suggestions, keep them active in the session history so they remain
            # in memory, but only display them visually when should_suggest_products is True.
            if previous_suggested:
                # Retrieve the full actual SuggestedProduct schemas list from history to avoid losing data
                suggested_products = []
                for msg in reversed(state.messages):
                    if msg.role == "assistant" and msg.suggested_products:
                        suggested_products = msg.suggested_products
                        break
            elif should_suggest_products:
                logger.info(f"Generating semantic recommendations for user_id {user_id}...")
                retrieval_query = json.dumps(
                    {"profile": state.profile.model_dump(), "latest_user_message": user_message},
                    ensure_ascii=False,
                )
                try:
                    suggested_products = await asyncio.to_thread(
                        semantic_search, retrieval_query, limit=3
                    )
                    logger.info(f"Retrieved {len(suggested_products)} products: {[p.product for p in suggested_products]}")
                except Exception as err:
                    logger.error(f"Semantic search retrieval failed: {err}", exc_info=True)
                    suggested_products = []
            else:
                suggested_products = []

            jargon_result, traps_result = await run_enrichment(
                state.profile,
                user_message,
                advisor_turn.assistant_message,
                advisor_turn.jargon_terms,
                is_final_recommendation,
            )
            assistant_message = advisor_turn.assistant_message
            jargon = jargon_result.explanations
            traps = traps_result.warnings
            # If a user is asking follow-up questions, we keep their checklist context visible dynamically
            # by fetching previous recommendations if present, while trusting advisor_turn.recommendation if new.
            recommendation = advisor_turn.recommendation
            if recommendation is None:
                # Find previous recommendation in history to keep it rendered
                for msg in reversed(state.messages):
                    if msg.role == "assistant" and msg.recommendation:
                        recommendation = msg.recommendation
                        break

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
            jargon_only=advisor_turn.jargon_only,
            warnings=traps,
            recommendation=recommendation,
            suggested_products=suggested_products,
            should_suggest_products=should_suggest_products,
        )
    )
    await sessions.update(state)
    return ChatResponse(
        user_id=user_id,
        assistant_message=assistant_message,
        profile=state.profile,
        jargon=jargon,
        jargon_only=advisor_turn.jargon_only,
        warnings=traps,
        recommendation=recommendation,
        suggested_products=suggested_products,
        should_suggest_products=should_suggest_products,
    )


def _apply_profile_update(profile: UserProfile, updates: dict[str, object]) -> None:
    for field_name, value in updates.items():
        if value is not None:
            setattr(profile, field_name, value)
