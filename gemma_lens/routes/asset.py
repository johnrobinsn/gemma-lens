"""GET /asset/{modality}/{id} — serve the raw image or audio file."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from gemma_lens import config, storage

router = APIRouter()


@router.get("/asset/image/{image_id}")
async def get_image(image_id: int, request: Request) -> FileResponse:
    with storage.connect() as conn:
        meta = storage.get_image(conn, image_id)
    if not meta:
        raise HTTPException(404, "image not found")
    path = config.COCO_IMAGES_DIR / meta["filename"]
    if not path.exists():
        raise HTTPException(404, "image file missing on disk")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/asset/audio/{audio_id}")
async def get_audio(audio_id: int, request: Request) -> FileResponse:
    with storage.connect() as conn:
        meta = storage.get_audio(conn, audio_id)
    if not meta:
        raise HTTPException(404, "audio not found")
    path = config.ESC50_AUDIO_DIR / meta["filename"]
    if not path.exists():
        raise HTTPException(404, "audio file missing on disk")
    return FileResponse(path, media_type="audio/wav")
