"""FastAPI application factory. Loads model + index once at startup."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rich.console import Console

from gemma_lens import config
from gemma_lens.index import SearchIndex
from gemma_lens.model import EmbeddingModel
from gemma_lens.routes import asset, examples, search

console = Console()
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    console.log("Loading EmbeddingGemma 2 …")
    app.state.model = EmbeddingModel()
    console.log(f"Model on {app.state.model.device} ({app.state.model.dtype})")
    console.log(f"Loading index from {config.INDEX_PATH}")
    app.state.index = SearchIndex()
    console.log(
        f"Index ready: {len(app.state.index.image_ids)} images, "
        f"{len(app.state.index.audio_ids)} audio"
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Gemma-Lens", lifespan=lifespan)

    app.include_router(search.router)
    app.include_router(asset.router)
    app.include_router(examples.router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        async def index_html() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app
