# Ask Insurance

A concise hackathon POC that helps people identify the health-insurance cover
they should look for. It uses the OpenAI Agents SDK with Gemini through
Google's OpenAI-compatible API, FastAPI for the backend, and Streamlit for the
chat interface.

## What it does

- Guides the user through the information needed to choose health insurance.
- Explains insurance jargon in plain language, always with an example.
- Flags common policy traps relevant to the user's needs.
- Produces a practical buying checklist, not a named-product recommendation.

Session memory is keyed by `user_id` and held in FastAPI process memory. It
survives Streamlit reruns, but it is intentionally lost when the backend
restarts. This POC has no database or authentication.

## Setup

Use Python 3.12 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Google AI Studio key to `.env`:

```env
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash
```

## Run

Start the API in one terminal:

```bash
uvicorn backend.main:app --reload
```

Start the interface in another:

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL shown in the terminal, usually
`http://localhost:8501`.

## Scope

This is educational guidance, not medical, legal, or financial advice. It does
not compare live policy products, quote premiums, or guarantee claim outcomes.
Final coverage depends on the policy wording and insurer underwriting.

# Deploying the backend to Render

This repository includes a `render.yaml` Blueprint configuration. When creating a
Render web service, use the following commands if you configure it manually:

```
Build Command: pip install -r requirements.txt
Start Command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
Health Check Path: /health
```

Do not use `--reload` in Render. Render supplies the `PORT` environment variable
and the server must listen on `0.0.0.0` so that it is reachable outside the
container.
