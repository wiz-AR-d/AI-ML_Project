# AI-ML Project — FastAPI Backend

A clean, production-ready FastAPI backend for the AI/ML Project.

## Project Structure

```
backend/
├── app/
│   ├── core/
│   │   └── config.py       # Pydantic settings (env vars)
│   ├── routers/
│   │   ├── health.py       # GET /api/v1/health
│   │   └── predict.py      # POST /api/v1/ml/predict
│   └── main.py             # FastAPI app factory
├── main.py                 # Uvicorn entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment variables
cp .env.example .env

# 4. Run the dev server
python main.py
# OR
uvicorn app.main:app --reload
```

The API will be available at **http://localhost:8000**

## API Docs

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Swagger UI (interactive) |
| http://localhost:8000/redoc | ReDoc documentation |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root welcome message |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/ml/predict` | ML prediction (placeholder) |

## Adding ML Models

Edit `app/routers/predict.py` and replace the placeholder logic with your actual model inference code.
