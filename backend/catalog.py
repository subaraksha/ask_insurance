"""MongoDB-backed policy wording ingestion and semantic product retrieval."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure, PyMongoError
from sentence_transformers import SentenceTransformer

from backend.schemas import CatalogProduct, IngestionItemResult, SuggestedProduct

load_dotenv()

DATABASE_NAME = os.getenv("MONGODB_DATABASE", "ask_insurance")
PRODUCTS_COLLECTION = "insurance_products"
CHUNKS_COLLECTION = "policy_chunks"
VECTOR_INDEX_NAME = "policy_vector_index"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = 384
MAX_PDF_BYTES = 30 * 1024 * 1024
MAX_PDF_PAGES = 500


class CatalogConfigurationError(RuntimeError):
    """Raised when MongoDB or the embedding model has not been configured."""


def _now() -> datetime:
    return datetime.now(UTC)


@lru_cache
def get_database():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise CatalogConfigurationError("MONGODB_URI is not configured. Add it to .env.")
    client = MongoClient(uri, serverSelectionTimeoutMS=8_000, appname="ask-insurance")
    return client[DATABASE_NAME]


@lru_cache
def get_embedder() -> SentenceTransformer:
    """Load a local, open-source model. It is downloaded once on first use."""
    return SentenceTransformer(EMBEDDING_MODEL)


def _product_id(product: CatalogProduct) -> str:
    stable_value = f"{product.insurance_company}|{product.product}".lower()
    return hashlib.sha256(stable_value.encode()).hexdigest()[:24]


def _validate_source(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("pdf_source must be a public http(s) URL.")


def _download_pdf(url: str) -> bytes:
    _validate_source(url)
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
    )
    with urlopen(request, timeout=45) as response:  # nosec B310: URL is validated above
        data = response.read(MAX_PDF_BYTES + 1)

    if len(data) > MAX_PDF_BYTES:
        raise ValueError("Policy wording is larger than the 30 MB ingestion limit.")
    if not data.startswith(b"%PDF"):
        raise ValueError("The policy wording URL did not return a PDF file.")
    return data


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_text = []
    for page in reader.pages[:MAX_PDF_PAGES]:
        text = page.extract_text(extraction_mode="layout") or ""
        if text.strip():
            page_text.append(text)
    extracted = "\n".join(page_text)
    if not extracted.strip():
        raise ValueError("No readable text was found in the policy PDF (it may be scanned).")
    return re.sub(r"[ \t]+", " ", extracted).strip()


def _chunk_text(text: str, chunk_size: int = 280, overlap: int = 45) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def _embedding_input(product: CatalogProduct, excerpt: str) -> str:
    return (
        f"Health insurance product: {product.product}. Insurer: {product.insurance_company}. "
        f"Best suited for: {product.primary_use_case}. Policy wording: {excerpt}"
    )


def _embeddings(inputs: list[str]) -> list[list[float]]:
    if not inputs:
        return []
    values = get_embedder().encode(
        inputs,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16,
    )
    return [value.tolist() for value in values]


def ensure_database_setup() -> dict[str, str]:
    """Create ordinary indexes and request the Atlas Vector Search index if available."""
    database = get_database()
    products = database[PRODUCTS_COLLECTION]
    chunks = database[CHUNKS_COLLECTION]
    products.create_index([( "product_key", ASCENDING)], unique=True, name="product_key_unique")
    products.create_index([( "insurance_company", ASCENDING), ("product", ASCENDING)])
    chunks.create_index([( "product_id", ASCENDING), ("chunk_number", ASCENDING)], unique=True)
    chunks.create_index(
        [("text", "text"), ("primary_use_case", "text"), ("product", "text")],
        name="policy_text_fallback",
    )

    definition = {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": EMBEDDING_DIMENSIONS, "similarity": "cosine"},
            {"type": "filter", "path": "insurance_company"},
            {"type": "filter", "path": "primary_use_case"},
        ]
    }
    try:
        database.command(
            {"createSearchIndexes": CHUNKS_COLLECTION, "indexes": [{"name": VECTOR_INDEX_NAME, "definition": definition}]}
        )
        vector_index = "requested"
    except OperationFailure as error:
        # Atlas returns an error when the index already exists; attempting an update is safe.
        if "already exists" in str(error).lower():
            try:
                database.command(
                    {"updateSearchIndex": CHUNKS_COLLECTION, "name": VECTOR_INDEX_NAME, "definition": definition}
                )
                vector_index = "updating"
            except OperationFailure:
                vector_index = "exists"
        else:
            # A standard Mongo deployment has no Atlas Search. Text fallback remains usable.
            vector_index = "unavailable"
    return {"database": DATABASE_NAME, "vector_index": vector_index}


def ingest_product(product: CatalogProduct, *, fetch_policy_wording: bool = True) -> IngestionItemResult:
    database = get_database()
    product_id = _product_id(product)
    product_key = f"{product.insurance_company}|{product.product}".lower()
    source_error: str | None = None
    wording_text = ""

    if fetch_policy_wording:
        try:
            wording_text = _extract_pdf_text(_download_pdf(str(product.pdf_source)))
        except Exception as error:  # keep the supplied public source searchable even when it fails
            source_error = str(error)

    summary = _embedding_input(product, "Official policy wording and product metadata.")
    text_chunks = [summary] + _chunk_text(wording_text)
    embeddings = _embeddings(text_chunks)
    now = _now()
    document = {
        "_id": product_id,
        "product_key": product_key,
        "insurance_company": product.insurance_company,
        "product": product.product,
        "primary_use_case": product.primary_use_case,
        "pdf_source": str(product.pdf_source),
        "source_status": "indexed" if wording_text else "metadata_only",
        "source_error": source_error,
        "chunk_count": len(text_chunks),
        "updated_at": now,
    }
    database[PRODUCTS_COLLECTION].replace_one({"_id": product_id}, document, upsert=True)
    chunks = database[CHUNKS_COLLECTION]
    chunks.delete_many({"product_id": product_id})
    if text_chunks:
        chunks.insert_many(
            [
                {
                    "_id": f"{product_id}:{index}",
                    "product_id": product_id,
                    "chunk_number": index,
                    "text": text,
                    "embedding": embedding,
                    "insurance_company": product.insurance_company,
                    "product": product.product,
                    "primary_use_case": product.primary_use_case,
                    "pdf_source": str(product.pdf_source),
                    "created_at": now,
                }
                for index, (text, embedding) in enumerate(zip(text_chunks, embeddings, strict=True))
            ]
        )
    return IngestionItemResult(
        insurance_company=product.insurance_company,
        product=product.product,
        chunks_indexed=len(text_chunks),
        status="indexed" if wording_text else "metadata_only",
        error=source_error,
    )


def ingest_products(products: list[CatalogProduct], *, fetch_policy_wording: bool = True) -> list[IngestionItemResult]:
    ensure_database_setup()
    results: list[IngestionItemResult] = []
    for product in products:
        try:
            results.append(ingest_product(product, fetch_policy_wording=fetch_policy_wording))
        except Exception as error:
            results.append(
                IngestionItemResult(
                    insurance_company=product.insurance_company,
                    product=product.product,
                    chunks_indexed=0,
                    status="failed",
                    error=str(error),
                )
            )
    return results


def _to_suggestions(rows: list[dict[str, Any]], limit: int) -> list[SuggestedProduct]:
    suggestions: list[SuggestedProduct] = []
    seen: set[str] = set()
    for row in rows:
        product_id = row["product_id"]
        if product_id in seen:
            continue
        seen.add(product_id)
        suggestions.append(
            SuggestedProduct(
                product_id=product_id,
                insurance_company=row["insurance_company"],
                product=row["product"],
                primary_use_case=row["primary_use_case"],
                pdf_source=row["pdf_source"],
                relevance_score=round(float(row.get("score", 0)), 3),
                evidence_excerpt=row.get("text", "")[:600],
            )
        )
        if len(suggestions) >= limit:
            break
    return suggestions


def _local_vector_fallback(
    collection: Collection[dict[str, Any]], query_embedding: list[float], query: str, limit: int
) -> list[SuggestedProduct]:
    """Small-catalog fallback for deployments without Atlas Vector Search.

    The vectors remain persisted in MongoDB; cosine scoring happens in-process
    only until the deployment can serve the Atlas `$vectorSearch` stage.
    """
    rows = list(collection.find({}, {"_id": 0}))
    for row in rows:
        embedding = row.pop("embedding", [])
        row["score"] = sum(left * right for left, right in zip(query_embedding, embedding, strict=True))
    _apply_use_case_rerank(rows, query)
    rows.sort(key=lambda row: row["score"], reverse=True)
    return _to_suggestions(rows, limit)


def _apply_use_case_rerank(rows: list[dict[str, Any]], query: str) -> None:
    """Keep broad vector matches aligned with the catalog's explicit use-case label."""
    query_lower = query.lower()
    for row in rows:
        use_case = row["primary_use_case"].lower()
        boost = 0.0
        if any(term in query_lower for term in ("senior", "elder", "retired", "60 year", "65 year")):
            boost += 0.55 if "senior" in use_case else 0.0
        elif "senior" in use_case:
            boost -= 0.18
        if any(term in query_lower for term in ("family", "spouse", "children", "child", "parents")):
            boost += 0.5 if "family" in use_case else 0.0
        if any(term in query_lower for term in ("top-up", "top up", "deductible")):
            boost += 0.55 if "top-up" in use_case else 0.0
        if any(term in query_lower for term in ("heart", "cardiac")):
            boost += 0.65 if "cardiac" in use_case else 0.0
        elif "cardiac" in use_case:
            boost -= 0.18
        if any(term in query_lower for term in ("critical illness", "cancer", "stroke")):
            boost += 0.55 if "critical illness" in use_case else 0.0
        elif "critical illness" in use_case:
            boost -= 0.22
        if any(term in query_lower for term in ("budget", "₹", "rs.", "rupee", "premium")):
            boost += 0.42 if "budget" in use_case else 0.0
        if any(term in query_lower for term in ("first", "healthy", "individual", "comprehensive")):
            boost += 0.25 if ("comprehensive" in use_case or "individual" in use_case) else 0.0
        row["score"] = float(row.get("score", 0)) + boost


def _build_natural_language_query(query: str) -> str:
    """Parse JSON queries from conversation profiles and build clear natural language sentences."""
    try:
        data = json.loads(query)
        if isinstance(data, dict) and "profile" in data:
            profile = data["profile"]
            latest_msg = data.get("latest_user_message", "")
            
            parts = ["Health insurance policy recommendation query."]
            
            cov_for = profile.get("coverage_for")
            if cov_for:
                parts.append(f"Coverage needed for: {cov_for}.")
                if any(term in str(cov_for).lower() for term in ("parent", "senior", "elder", "grandfather", "grandmother", "mother", "father")):
                    parts.append("This is for senior citizens / elderly people.")
                
            ages = profile.get("ages", [])
            if ages:
                parts.append(f"Ages of people to cover: {', '.join(map(str, ages))} years old.")
                if any(age >= 60 for age in ages):
                    parts.append("This is for senior citizens / elderly people.")
                    
            city = profile.get("city")
            if city:
                parts.append(f"Location / City: {city}.")
                
            sum_insured = profile.get("sum_insured")
            if sum_insured:
                parts.append(f"Sum insured: {sum_insured}.")
                
            med_conds = profile.get("medical_conditions", [])
            if med_conds:
                parts.append(f"Pre-existing medical conditions: {', '.join(med_conds)}.")
                
            cov_prefs = profile.get("coverage_preferences", [])
            if cov_prefs:
                parts.append(f"Preferences: {', '.join(cov_prefs)}.")
                
            hosp_pref = profile.get("hospital_preference")
            if hosp_pref:
                parts.append(f"Hospital preference: {hosp_pref}.")
                
            if latest_msg:
                parts.append(f"User context: {latest_msg}")
                
            return " ".join(parts)
    except Exception:
        pass
    return query


def semantic_search(query: str, *, limit: int = 3) -> list[SuggestedProduct]:
    database = get_database()
    nl_query = _build_natural_language_query(query)
    query_embedding = _embeddings([nl_query])[0]
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max(limit * 30, 60),
                "limit": max(limit * 5, 12),
            }
        },
        {"$project": {"_id": 0, "product_id": 1, "insurance_company": 1, "product": 1, "primary_use_case": 1, "pdf_source": 1, "text": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    try:
        rows = list(database[CHUNKS_COLLECTION].aggregate(pipeline))
        _apply_use_case_rerank(rows, nl_query)
        suggestions = _to_suggestions(rows, limit)
        if suggestions:
            return suggestions
    except (OperationFailure, PyMongoError):
        pass

    local_suggestions = _local_vector_fallback(
        database[CHUNKS_COLLECTION], query_embedding, nl_query, limit
    )
    if local_suggestions:
        return local_suggestions

    # A metadata-only database without vectors can still serve a keyword fallback.
    words = [word for word in re.findall(r"[a-zA-Z]{3,}", nl_query.lower()) if word not in {"health", "insurance", "policy", "cover", "looking", "first"}]
    query_filter = {"$or": [{"primary_use_case": {"$regex": word, "$options": "i"}} for word in words]}
    if not words:
        query_filter = {}
    rows = list(database[CHUNKS_COLLECTION].find(query_filter, {"embedding": 0}).limit(max(limit * 8, 20)))
    for row in rows:
        haystack = f"{row['product']} {row['primary_use_case']} {row['text'][:400]}".lower()
        row["score"] = sum(word in haystack for word in words) / max(len(words), 1)
    _apply_use_case_rerank(rows, nl_query)
    rows.sort(key=lambda row: row["score"], reverse=True)
    return _to_suggestions(rows, limit)


def get_policy_context(product_id: str, *, max_chunks: int = 8) -> dict[str, Any] | None:
    """Return the most decision-relevant wording excerpts for an LLM-grounded view."""
    database = get_database()
    product = database[PRODUCTS_COLLECTION].find_one({"_id": product_id}, {"_id": 0})
    if product is None:
        return None
    chunks = list(
        database[CHUNKS_COLLECTION]
        .find({"product_id": product_id}, {"_id": 0, "embedding": 0})
        .sort("chunk_number", ASCENDING)
    )
    wording_chunks = [chunk for chunk in chunks if chunk.get("chunk_number") != 0]
    keywords = (
        "cover", "benefit", "hospital", "exclusion", "not covered", "waiting period",
        "pre-existing", "co-pay", "copay", "deductible", "room rent", "sub-limit",
        "restoration", "renewal", "sum insured",
    )
    ranked = sorted(
        wording_chunks,
        key=lambda chunk: sum(keyword in chunk["text"].lower() for keyword in keywords),
        reverse=True,
    )
    selected = ranked[:max_chunks]
    if not selected and chunks:
        selected = chunks[:1]
    context = "\n\n".join(
        f"[Policy excerpt {index + 1}]\n{chunk['text'][:2_500]}"
        for index, chunk in enumerate(selected)
    )
    return {
        "product_id": product_id,
        "insurance_company": product["insurance_company"],
        "product": product["product"],
        "primary_use_case": product["primary_use_case"],
        "pdf_source": product["pdf_source"],
        "source_status": "wording_available" if wording_chunks else "metadata_only",
        "policy_context": context,
    }


def catalog_status() -> dict[str, Any]:
    database = get_database()
    return {
        "database": DATABASE_NAME,
        "products": database[PRODUCTS_COLLECTION].count_documents({}),
        "chunks": database[CHUNKS_COLLECTION].count_documents({}),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "vector_index": VECTOR_INDEX_NAME,
    }
