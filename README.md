# Ask Insurance

Ask Insurance is a health-insurance research assistant for India. It combines a
FastAPI API, Streamlit interface, MongoDB product store, Atlas Vector Search,
and a locally-run open-source embedding model. It is educational only: the
app links to official wordings and never treats a retrieval match as a quote,
claim guarantee, or final buying recommendation.

## What changed

- `insurance_products` stores each insurer/product/use-case/official wording URL.
- `policy_chunks` stores the extracted policy text in chunks and a 384-dimension
  vector generated locally using `sentence-transformers/all-MiniLM-L6-v2`.
- The supplied 25-product catalog is in `data/product_catalog.json`.
- `POST /catalog/ingest` accepts future catalog records; it upserts products,
  downloads the public PDF, extracts readable text, chunks it, embeds it and
  indexes it.
- `POST /catalog/seed` ingests the supplied catalog.
- `GET /catalog/search` retrieves the best matching products. The chat calls it
  automatically and the UI presents selectable product cards and a metadata
  comparison table.

## Setup

Use Python 3.12 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set these values in `.env` (your existing `MONGODB_URI` is used as-is):

```env
MONGODB_URI=mongodb+srv://...
MONGODB_DATABASE=ask_insurance
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash
# Optional but recommended for production catalog writes
INGEST_API_KEY=choose-a-long-random-secret
# Optional; this is the default local open-source model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The first catalog ingestion downloads the model files and public PDFs. The
embeddings are generated on the application machine; no product text is sent
to an embedding API.

## Run and ingest

Start the API:

```bash
uvicorn backend.main:app --reload
```

In a second terminal, run the UI:

```bash
streamlit run frontend/app.py
```

Ingest the provided catalog (downloads wordings, so this can take a few
minutes):

```bash
curl -X POST 'http://127.0.0.1:8000/catalog/seed' \
  -H 'X-Admin-Key: your-ingest-key'
```

For a quick metadata/vector smoke test that skips downloading PDFs:

```bash
curl -X POST 'http://127.0.0.1:8000/catalog/seed?fetch_policy_wordings=false' \
  -H 'X-Admin-Key: your-ingest-key'
```

When `INGEST_API_KEY` is not set, the header is not required for local
development. Do set it before exposing the service publicly.

## Add or refresh products

Send one or more public policy wording URLs to the ingestion endpoint. A
repeat record is safe: it replaces that product's old chunks.

```bash
curl -X POST 'http://127.0.0.1:8000/catalog/ingest' \
  -H 'Content-Type: application/json' \
  -H 'X-Admin-Key: your-ingest-key' \
  -d '{
    "products": [{
      "insurance_company": "Example Insurer",
      "product": "Example Health Plan",
      "primary_use_case": "Comprehensive Individual",
      "pdf_source": "https://example.com/policy-wording.pdf"
    }]
  }'
```

Check ingestion and Mongo state with `GET /catalog/status`; test retrieval with
`GET /catalog/search?query=healthy%2028%20year%20old%20first%20health%20cover`.

## Atlas Vector Search index

On API startup and before each ingestion, the service requests this index via
MongoDB's `createSearchIndexes` command. Atlas builds it asynchronously; while
it builds (or when using standard MongoDB), the API uses a simple keyword
fallback. The required index is named `policy_vector_index` on the
`policy_chunks` collection:

```json
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
    {"type": "filter", "path": "insurance_company"},
    {"type": "filter", "path": "primary_use_case"}
  ]
}
```

If the connected Mongo deployment does not permit Search-index commands, create
that JSON definition manually in Atlas Search. A standard local MongoDB server
does not provide `$vectorSearch`; the app remains usable with its fallback,
but Atlas Vector Search is required for semantic retrieval.

## Deployment

The included `render.yaml` installs `requirements.txt` and starts the API. For
Render, configure `MONGODB_URI`, `GOOGLE_API_KEY`, and `INGEST_API_KEY` as
secret environment variables. Use persistent model caching or a worker image
if cold-start download time becomes important.
