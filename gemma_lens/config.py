"""Paths, model id, device + dtype selection."""

from __future__ import annotations

from pathlib import Path

import torch

MODEL_ID = "google/embeddinggemma-2"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# COCO val2017 — unpacked naturally from the zip
COCO_IMAGES_DIR = DATA_DIR / "images" / "val2017"
COCO_CAPTIONS = DATA_DIR / "annotations" / "captions_val2017.json"

# ESC-50 — unpacked under data/esc50/ (ESC-50-master/ prefix stripped)
ESC50_DIR = DATA_DIR / "esc50"
ESC50_AUDIO_DIR = ESC50_DIR / "audio"
ESC50_META = ESC50_DIR / "meta" / "esc50.csv"

# Metadata + embeddings
DB_PATH = DATA_DIR / "gemma_lens.db"
INDEX_PATH = DATA_DIR / "index.npz"

# Dataset download URLs (verified 2026-10-07)
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

# Corpus limits (keep ingest tractable; spec §3.2)
NUM_IMAGES = 5000  # COCO val2017 subset

# Model config
AUDIO_SR = 16000  # EmbeddingGemma 2 expects 16 kHz mono
DEFAULT_K = 16


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """Choose dtype. Reject float16 — EmbeddingGemma 2 silently returns NaN in fp16."""
    if requested == "float16" or requested is torch.float16:
        raise ValueError(
            "float16 produces silent NaN with EmbeddingGemma 2. "
            "Use 'bfloat16' or 'float32'."
        )
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    # auto
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32
