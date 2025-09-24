from fastapi import FastAPI
from contextlib import asynccontextmanager
from .utils.load_model import load_model_from_mlflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_from_mlflow()
    print("Model loaded and ready to use.")
    yield


version = "v1"

version_prefix = "/api/{version}"

description = """..."""

app = FastAPI(
    title="Rotten Bot Inference API",
    description=description,
    version=version,
    license_info={"name": "MIT License", "url": "https://opensource.org/license/mit"},
    contact={
        "name": "Niels Scholz",
    },
    terms_of_service="httpS://example.com/tos",
    openapi_url=f"{version_prefix}/openapi.json",
    docs_url=f"{version_prefix}/docs",
    redoc_url=f"{version_prefix}/redoc",
    lifespan=lifespan,
)

# register_error_handlers(app)
# register_middleware(app)


# app.include_router(book_router, prefix=f"{version_prefix}/books", tags=["books"])
# app.include_router(auth_router, prefix=f"{version_prefix}/auth", tags=["auth"])
