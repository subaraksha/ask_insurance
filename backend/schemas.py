from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class UserProfile(BaseModel):
    country: str = "India"
    city: str | None = None
    coverage_for: str | None = None
    ages: list[int] = Field(default_factory=list)
    family_members: list[str] = Field(default_factory=list)
    medical_conditions: list[str] = Field(default_factory=list)
    current_cover: str | None = None
    budget: str | None = None
    sum_insured: str | None = None
    hospital_preference: str | None = None
    coverage_preferences: list[str] = Field(default_factory=list)


class ProfileUpdate(BaseModel):
    country: str | None = None
    city: str | None = None
    coverage_for: str | None = None
    ages: list[int] | None = None
    family_members: list[str] | None = None
    medical_conditions: list[str] | None = None
    current_cover: str | None = None
    budget: str | None = None
    sum_insured: str | None = None
    hospital_preference: str | None = None
    coverage_preferences: list[str] | None = None


class Recommendation(BaseModel):
    summary: str
    suggested_policy_structure: str
    target_coverage: str
    must_have_features: list[str] = Field(min_length=1)
    optional_features: list[str] = Field(default_factory=list)
    avoid_or_verify: list[str] = Field(min_length=1)
    questions_for_insurer: list[str] = Field(min_length=1)


class CatalogProduct(BaseModel):
    """Public product metadata plus its authoritative policy-wording URL."""

    insurance_company: str = Field(min_length=1, max_length=120)
    product: str = Field(min_length=1, max_length=160)
    primary_use_case: str = Field(min_length=1, max_length=160)
    pdf_source: AnyHttpUrl


class IngestCatalogRequest(BaseModel):
    products: list[CatalogProduct] = Field(min_length=1, max_length=100)
    fetch_policy_wordings: bool = True


class IngestionItemResult(BaseModel):
    insurance_company: str
    product: str
    chunks_indexed: int
    status: Literal["indexed", "metadata_only", "failed"]
    error: str | None = None


class IngestionResponse(BaseModel):
    results: list[IngestionItemResult]
    indexed: int
    metadata_only: int
    failed: int


class SuggestedProduct(BaseModel):
    product_id: str
    insurance_company: str
    product: str
    primary_use_case: str
    pdf_source: AnyHttpUrl
    relevance_score: float = 0
    evidence_excerpt: str = ""


class PolicyInsight(BaseModel):
    product_id: str
    insurance_company: str
    product: str
    source_status: Literal["wording_available", "metadata_only"]
    overview: str
    key_covers: list[str] = Field(default_factory=list)
    key_exclusions: list[str] = Field(default_factory=list)
    waiting_periods: list[str] = Field(default_factory=list)
    limits_or_cost_sharing: list[str] = Field(default_factory=list)
    important_checks: list[str] = Field(default_factory=list)
    source_note: str
    pdf_source: AnyHttpUrl


class CompareProductsRequest(BaseModel):
    product_ids: list[str] = Field(min_length=2, max_length=4)


class ComparisonRow(BaseModel):
    criterion: str
    values: list[str]


class PolicyComparison(BaseModel):
    product_ids: list[str] = Field(min_length=2)
    product_labels: list[str] = Field(min_length=2)
    rows: list[ComparisonRow] = Field(min_length=1)
    important_notes: list[str] = Field(default_factory=list)
    source_note: str


class AdvisorTurn(BaseModel):
    assistant_message: str = Field(min_length=1)
    scope_status: Literal["in_scope", "out_of_scope"] = "in_scope"
    profile_updates: ProfileUpdate = Field(default_factory=ProfileUpdate)
    jargon_terms: list[str] = Field(default_factory=list)
    jargon_only: bool = False
    show_buying_checks: bool = False
    ready_for_recommendation: bool = False
    recommendation: Recommendation | None = None


class JargonExplanation(BaseModel):
    term: str
    simple_meaning: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    example: str = Field(min_length=1)
    what_to_prefer: str = Field(min_length=1)


class JargonResult(BaseModel):
    explanations: list[JargonExplanation] = Field(default_factory=list)


class TrapWarning(BaseModel):
    title: str
    severity: Literal["high", "medium", "low"]
    why_it_matters: str
    what_to_check: str


class TrapResult(BaseModel):
    warnings: list[TrapWarning] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)
    jargon: list[JargonExplanation] = Field(default_factory=list)
    jargon_only: bool = False
    warnings: list[TrapWarning] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    suggested_products: list[SuggestedProduct] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SessionState(BaseModel):
    user_id: str
    profile: UserProfile = Field(default_factory=UserProfile)
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)


class SessionResponse(BaseModel):
    user_id: str
    profile: UserProfile
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    user_id: str
    assistant_message: str
    profile: UserProfile
    jargon: list[JargonExplanation]
    jargon_only: bool = False
    warnings: list[TrapWarning]
    recommendation: Recommendation | None = None
    suggested_products: list[SuggestedProduct] = Field(default_factory=list)
