from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import health, analyze

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="AI/ML Project REST API built with FastAPI",
        version="0.1.0",
        debug=settings.DEBUG,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, prefix=settings.API_V1_PREFIX, tags=["Health"])

    app.include_router(
        analyze.router, prefix=f"{settings.API_V1_PREFIX}/ml", tags=["NLP"]
    )

    @app.on_event("startup")
    async def load_nlp_data():
        """Load arXiv dataset and build TF-IDF corpus at startup."""
        import asyncio
        from app.services.nlp_pipeline import load_dataset
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, load_dataset)

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": f"Welcome to {settings.APP_NAME}"}

    return app


app = create_app()
