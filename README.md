# 🔬 Pluto — NLP Research Analysis System

> **Milestone 1** · Traditional NLP pipeline for analyzing 5,000 arXiv research papers

[![Frontend](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel)](https://ai-ml-project-blue.vercel.app/)
[![Backend](https://img.shields.io/badge/Backend-Render-46E3B7?logo=render)](https://render.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)]()
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)]()

---

## 📖 Overview

Pluto is a full-stack NLP analysis system that lets you search and analyze scientific papers from the arXiv dataset. Enter keywords and the system finds matching papers, extracts key terms, discovers topic clusters, and generates extractive summaries — all using traditional NLP techniques (no LLMs).

### ✨ Features

- **🔍 Keyword Search** — Search through 5,000 arXiv paper abstracts
- **📊 TF-IDF Key Terms** — Extract important terms using corpus-level TF-IDF scoring
- **🗂️ Topic Clustering** — Discover themes via Latent Dirichlet Allocation (LDA)
- **📝 Extractive Summarization** — Generate summaries by scoring sentences with TF-IDF
- **⚡ Real-time Analysis** — Pre-loaded corpus for instant results

---

## 🧠 NLP Pipeline

The backend implements a traditional NLP pipeline adapted from a Colab notebook:

```
Raw Abstract → Clean Text → Tokenize → Remove Stopwords → Lemmatize → TF-IDF Matrix
                                                                          ↓
                                                              Keyword Search
                                                              Key Term Extraction
                                                              LDA Topic Clustering
                                                              Extractive Summarization
```

| Step | Method | Library |
|------|--------|---------|
| Text Cleaning | Lowercase + punctuation removal | Python `string` |
| Tokenization | Whitespace split | Built-in |
| Stop-word Removal | NLTK English stop words | `nltk` |
| Lemmatization | WordNet Lemmatizer | `nltk` |
| TF-IDF | Corpus-level vectorization | `scikit-learn` |
| Topic Modeling | Latent Dirichlet Allocation | `scikit-learn` |
| Summarization | TF-IDF sentence scoring (extractive) | Custom |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, Vite, Vanilla CSS |
| **Backend** | FastAPI, Uvicorn, Python 3.9+ |
| **NLP** | NLTK, scikit-learn, NumPy, Pandas |
| **Dataset** | arXiv (5,000 papers, bundled as JSON) |
| **Deployment** | Vercel (frontend), Render (backend) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
python main.py
```

The server starts at `http://localhost:8000`. On first run, it loads 20k arXiv papers and builds the TF-IDF matrix (~10-15s).

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`. The Vite proxy forwards `/api` requests to the backend.

---

## 📁 Project Structure

```
AI-ML_Project/
├── README.md
├── render.yaml              # Render deployment blueprint
├── Procfile                 # Render start command
│
├── backend/
│   ├── main.py              # Uvicorn entry point
│   ├── requirements.txt
│   ├── build.sh             # Render build script
│   ├── data/
│   │   └── arxiv_20k.json   # Bundled arXiv dataset (20k papers)
│   └── app/
│       ├── main.py           # FastAPI app factory
│       ├── core/config.py    # Settings & CORS
│       ├── routers/
│       │   ├── analyze.py    # POST /api/v1/ml/analyze
│       │   └── health.py     # GET /api/v1/health
│       └── services/
│           └── nlp_pipeline.py  # Core NLP pipeline
│
└── frontend/
    ├── index.html
    ├── vite.config.js
    └── src/
        ├── App.jsx
        ├── App.css
        ├── index.css
        └── components/
            ├── InputPanel.jsx
            ├── OutputPanel.jsx
            ├── KeyTerms.jsx
            ├── TopicClusters.jsx
            └── Summary.jsx
```

---

## 🌐 Deployment

### Backend (Render)

1. Create a **Web Service** on [Render](https://render.com)
2. Connect GitHub repo → set **Root Directory** to `backend`
3. **Build Command:** `bash build.sh`
4. **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Deploy — no environment variables needed

### Frontend (Vercel)

1. Import repo on [Vercel](https://vercel.com)
2. Set **Root Directory** to `frontend`
3. Add env variable: `VITE_API_URL` = your Render backend URL
4. Deploy

---

## 📡 API

### `POST /api/v1/ml/analyze`

Analyze arXiv papers matching given keywords.

**Form Data:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `keywords` | `string[]` | required | Search terms |
| `num_topics` | `int` | `5` | Number of LDA topics |
| `summary_sentences` | `int` | `3` | Sentences per summary |

**Response:**
```json
{
  "terms": [{ "term": "neural", "score": 0.0842 }],
  "clusters": [{ "label": "Neural & Deep Learning", "keywords": [...], "doc_count": 12 }],
  "summary": ["Extractive summary sentence..."],
  "matched_papers": [{ "title": "...", "authors": "...", "categories": "..." }],
  "meta": { "doc_count": 150, "total_corpus": 20000, "elapsed_s": 1.23 }
}
```

---

## 📄 License

MIT
