"""FastAPI application factory and default application."""

from fastapi import FastAPI

from eris_ml.api.routes import health, model_info, prediction


def create_app() -> FastAPI:
    """Create an API without loading a model as an import side effect."""
    application = FastAPI(title="ERIS AI Retention Engine", version="0.1.0")
    application.state.model_bundle = None
    application.include_router(health.router)
    application.include_router(model_info.router)
    application.include_router(prediction.router)
    return application


app = create_app()
