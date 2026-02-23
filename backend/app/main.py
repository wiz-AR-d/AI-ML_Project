from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import health, predict, analyze

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
        predict.router, prefix=f"{settings.API_V1_PREFIX}/ml", tags=["ML"]
    )
    app.include_router(
        analyze.router, prefix=f"{settings.API_V1_PREFIX}/ml", tags=["NLP"]
    )

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": f"Welcome to {settings.APP_NAME}"}

    return app


app = create_app()
