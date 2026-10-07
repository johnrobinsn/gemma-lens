"""GET /examples — list of 'try these' preset queries for the UI."""

from __future__ import annotations

from fastapi import APIRouter

from gemma_lens.schemas import Example

router = APIRouter()

# Audio presets reference files in static/presets/. If absent, the UI should
# gracefully hide those buttons. v1 ships without pre-recorded audio clips;
# user records their own via mic.
EXAMPLES: list[Example] = [
    Example(label="red cars", modality="text", text="red cars"),
    Example(label="sunset over water", modality="text", text="sunset over water"),
    Example(label="snowy mountains", modality="text", text="snowy mountains"),
    Example(label="kitchen with food", modality="text", text="kitchen with food"),
    Example(label="white objects", modality="text", text="white objects"),
    Example(label="dogs playing", modality="text", text="dogs playing"),
    Example(label="cow sound", modality="audio", preset="cow_moo.wav"),
    Example(label="clapping", modality="audio", preset="clapping.wav"),
    Example(label="cat meow", modality="audio", preset="cat_meow.wav"),
    Example(label="rain", modality="audio", preset="rain.wav"),
]


@router.get("/examples", response_model=list[Example])
async def list_examples() -> list[Example]:
    return EXAMPLES
