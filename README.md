# 🩺 Ask Insurance

**Ask Insurance** is a highly sophisticated, AI-guided health-insurance buying assistant and research platform specifically designed for the complex health insurance market in India.

Instead of acting as a typical lead-generation tool or commission-driven policy seller, **Ask Insurance** acts as an **unbiased educational guide**. It empowers buyers by analyzing, comparing, and explaining complex policy wordings directly grounded in official, legally-binding policy documents.

---

## 🚀 Key Features

* **Adaptive Conversational Interview:** A supportive, low-friction AI dialogue that dynamically discovers buying criteria (family composition, ages, location, pre-existing medical conditions, features, and target budget) rather than forcing users through a rigid questionnaire.
* **Semantic Vector-Based Matching:** Uses local text embeddings to match a user's unique medical and demographic profile against an active, verified catalog of 25+ real-world insurance products.
* **LLM-Grounded Policy Insights:** Extracts and synthesizes crucial sections of legalese from the official policy PDFs—summarizing key covers, critical exclusions, waiting periods, sub-limits, and co-payment details with extreme accuracy.
* **Dynamic Side-by-Side Product Comparison:** Renders beautiful, fully responsive, and auto-formatted comparison matrices of selected policies directly inside the browser.
* **Jargon Buster (Explainability Engine):** Automatically detects complex insurance terms (e.g., *restoration benefits*, *co-pay*, *waiting periods*) in the advisor's responses, translating them into simplified layman definitions with practical mathematical examples.
* **100% Unbiased & Educational:** Always links directly back to the official insurer's hosted PDF, never invents terms, and emphasizes that final decisions belong to the insurer's official policy wording and underwriting.

---

## 🛠️ Technology Stack

* **Frontend:** [Streamlit](https://streamlit.io/) — Python-native, reactive, clean, and interactive single-page web interface.
* **Backend:** [FastAPI](https://fastapi.tiangolo.com/) — High-performance, asynchronous web framework providing endpoints for chat, search, comparison, and automated document ingestion.
* **Database & Vector Store:** [MongoDB Atlas](https://www.mongodb.com/atlas) — Stores structured product schemas and high-dimensional document chunk coordinates. Uses **Atlas Vector Search** (or standard vector fallback) to align user profiles with the policy database.
* **LLM Orchestration:** [Gemini 3.5 Flash](https://deepmind.google/technologies/gemini/) — Powers the adaptive insurance advisor, jargon explaining engine, and policy comparison reasoning.
* **Local Embeddings (Privacy-First):** `sentence-transformers/all-MiniLM-L6-v2` — Generates 384-dimensional dense vectors locally on-device. No user health details or raw product texts are ever shared with third-party embedding APIs.
* **PDF Extraction Engine:** `PyPDF` — Asynchronously processes and parses unstructured legally-binding PDF documents into clean, searchable, and chunkable text formats.

---

## 📂 Project Structure

```text
ask_insurance/
├── backend/
│   ├── agents.py       # Core LLM Agents (Advisor, Jargon Buster, Policy Analyst)
│   ├── catalog.py      # MongoDB Vector Store, local embedding pipeline, and search
│   ├── main.py         # FastAPI REST Router (Sessions, Catalog search & ingestion)
│   └── schemas.py      # Pydantic data schemas (Chat state, User profiles, Catalog)
├── frontend/
│   └── app.py          # Streamlit SPA Chat & Product comparison UI
├── data/
│   └── product_catalog.json   # Seed data containing 25+ premium real-world products
├── requirements.txt    # Python package dependencies
├── pyproject.toml      # Build systems and package configs
└── README.md           # This document!
```

---

## ⚙️ Setup & Local Installation

Use Python 3.12 or newer.

```bash
# Clone the repository
git clone https://github.com/subaraksha/ask_insurance.git
cd ask_insurance

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration template
cp .env.example .env
```

### Configure Environment Variables

Set these values in your `.env` file:

```env
MONGODB_URI=your_mongodb_atlas_connection_string
MONGODB_DATABASE=ask_insurance
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
INGEST_API_KEY=choose-a-long-random-secret  # To protect ingestion endpoints
```

---

## 🏃‍♂️ How to Run

### 1. Start the FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

### 2. Start the Streamlit Frontend (In a second terminal)

```bash
streamlit run frontend/app.py
```
