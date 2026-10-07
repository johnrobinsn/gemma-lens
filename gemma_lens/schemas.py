"""Pydantic request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from gemma_lens import config


class SearchRequest(BaseModel):
    modality: Literal["text", "image", "audio"]
    text: str | None = None
    image_b64: str | None = None
    audio_b64: str | None = None
    k: int = Field(default=config.DEFAULT_K, ge=1, le=64)


class SearchResult(BaseModel):
    id: int
    modality: Literal["image", "audio"]
    score: float
    filename: str
    url: str
    metadata: dict


class SearchResponse(BaseModel):
    query_ms: int
    embed_ms: int
    results: list[SearchResult]


class Example(BaseModel):
    label: str
    modality: Literal["text", "audio"]
    text: str | None = None
    preset: str | None = None
