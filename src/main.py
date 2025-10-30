import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI

from .config import Config
from .db.main import init_db
from .inference.routes import inference_router
from .inference.setup_observability import setup_observability
from .utils.load_model import load_classes_from_mlflow, load_model_from_mlflow

# path where the model and index_to_class.json will be stored
DST_PATH = "./model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    setup_observability("inference_service")

    if Config.USE_LOCAL:
        app.state.index_to_class = load_classes_from_mlflow(
            run_id="/app/index_to_class.json", dst_path=DST_PATH, use_local=True
        )

        app.state.model = load_model_from_mlflow(
            model_uri="/app/model", dst_path=DST_PATH, use_local=True
        )
    else:
        if not os.path.exists(DST_PATH):
            os.makedirs(DST_PATH)

        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

        app.state.index_to_class = load_classes_from_mlflow(
            run_id=Config.RUN_ID, dst_path=DST_PATH, use_local=Config.USE_LOCAL
        )

        app.state.model = load_model_from_mlflow(
            Config.MODEL_URI, DST_PATH, use_local=Config.USE_LOCAL
        )

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

app.include_router(
    inference_router, prefix=f"{version_prefix}/inference", tags=["inference"]
)
# app.include_router(book_router, prefix=f"{version_prefix}/books", tags=["books"])
# app.include_router(auth_router, prefix=f"{version_prefix}/auth", tags=["auth"])
