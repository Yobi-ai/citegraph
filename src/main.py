from fastapi import FastAPI

from .api.routes import router


def create_app() -> FastAPI:
    """Create and configure fastapi app.
    Returns:
    Configured fastapi application instance
    """
    app = FastAPI(
        title="citegraph-api",
        description="api for inferencing on new pdf",
        version="1.0.0",
    )

    # allowed_orgins: List[str] = []

    app.include_router(router)

    return app


app = create_app()
