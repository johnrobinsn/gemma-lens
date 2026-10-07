"""POST /search — embed query, retrieve top-K across image + audio corpora."""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from PIL import Image

from gemma_lens.schemas import SearchRequest, SearchResponse, SearchResult

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    model = request.app.state.model
    index = request.app.state.index

    t0 = time.perf_counter()
    if req.modality == "text":
        if not req.text:
            raise HTTPException(400, "text query requires `text` field")
        query = model.encode_text(req.text)
    elif req.modality == "image":
        if not req.image_b64:
            raise HTTPException(400, "image query requires `image_b64` field")
        img = _decode_image_b64(req.image_b64)
        query = model.encode_image(img)
    elif req.modality == "audio":
        if not req.audio_b64:
            raise HTTPException(400, "audio query requires `audio_b64` field")
        waveform, sr = _decode_audio_b64(req.audio_b64)
        query = model.encode_audio(waveform, sr)
    else:
        raise HTTPException(400, f"unknown modality {req.modality!r}")

    embed_ms = int((time.perf_counter() - t0) * 1000)
    results = index.search(query, k=req.k)
    query_ms = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(
        query_ms=query_ms,
        embed_ms=embed_ms,
        results=[SearchResult(**r) for r in results],
    )


def _decode_image_b64(b64: str) -> Image.Image:
    raw = base64.b64decode(_strip_data_url(b64))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _decode_audio_b64(b64: str) -> tuple[np.ndarray, int]:
    """Decode base64 audio (wav or webm). For webm, write to tmp + use librosa."""
    raw = base64.b64decode(_strip_data_url(b64))
    try:
        waveform, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return waveform, sr
    except Exception:
        pass

    # soundfile can't read opus-in-webm on all platforms. Fall back to librosa+tmp.
    import tempfile

    import librosa

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as f:
        f.write(raw)
        f.flush()
        waveform, sr = librosa.load(f.name, sr=None, mono=True)
    return np.asarray(waveform, dtype=np.float32), int(sr)


def _strip_data_url(b64: str) -> str:
    if "," in b64 and b64.startswith("data:"):
        return b64.split(",", 1)[1]
    return b64
