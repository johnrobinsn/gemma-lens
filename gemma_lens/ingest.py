"""One-shot ingestion: download datasets, build sqlite metadata, embed everything.

Idempotent — re-running skips any step whose output already exists on disk.

    uv run python -m gemma_lens.ingest
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from rich.console import Console
from tqdm import tqdm

from gemma_lens import config, storage

console = Console()
RNG = random.Random(42)


# ---------- Download helpers ----------

def download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        console.log(f"[dim]{label}: already downloaded ({dest.name})[/dim]")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    console.log(f"Downloading {label} → {dest}")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=label
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)


def extract_zip(zip_path: Path, out_dir: Path, strip_prefix: str | None = None) -> None:
    """Extract a zip. If strip_prefix is given, drop that leading folder from each entry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for m in tqdm(members, desc=f"extract {zip_path.name}", unit="file"):
            if m.endswith("/"):
                continue
            rel = m
            if strip_prefix and rel.startswith(strip_prefix):
                rel = rel[len(strip_prefix):]
            if not rel:
                continue
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


# ---------- Dataset setup ----------

def ensure_coco() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    images_zip = config.DATA_DIR / "val2017.zip"
    ann_zip = config.DATA_DIR / "annotations_trainval2017.zip"

    download(config.COCO_IMAGES_URL, images_zip, "COCO val2017 images")
    if not config.COCO_IMAGES_DIR.exists() or _count_dir(config.COCO_IMAGES_DIR, "*.jpg") < 100:
        extract_zip(images_zip, config.DATA_DIR / "images")

    download(config.COCO_ANNOTATIONS_URL, ann_zip, "COCO annotations")
    if not config.COCO_CAPTIONS.exists():
        extract_zip(ann_zip, config.DATA_DIR)


def ensure_esc50() -> None:
    zip_path = config.DATA_DIR / "esc50.zip"
    download(config.ESC50_URL, zip_path, "ESC-50")
    if not config.ESC50_META.exists():
        # Zip entries are prefixed with "ESC-50-master/" — strip and land at data/esc50/
        extract_zip(zip_path, config.ESC50_DIR, strip_prefix="ESC-50-master/")


def _count_dir(d: Path, glob: str) -> int:
    return sum(1 for _ in d.glob(glob))


# ---------- Metadata ingestion ----------

def build_metadata() -> None:
    storage.init_db()

    # COCO: pick a stable random subset of NUM_IMAGES, pair with first caption
    with open(config.COCO_CAPTIONS) as f:
        coco = json.load(f)
    captions_by_image: dict[int, str] = {}
    for ann in coco["annotations"]:
        captions_by_image.setdefault(ann["image_id"], ann["caption"])

    all_imgs = sorted(coco["images"], key=lambda x: x["id"])
    RNG.shuffle(all_imgs)
    sample = all_imgs[: config.NUM_IMAGES]

    with storage.connect() as conn:
        for img in tqdm(sample, desc="COCO metadata"):
            fname = img["file_name"]
            caption = captions_by_image.get(img["id"])
            storage.insert_image(conn, fname, caption)

    # ESC-50: read meta/esc50.csv
    with open(config.ESC50_META, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with storage.connect() as conn:
        for row in tqdm(rows, desc="ESC-50 metadata"):
            storage.insert_audio(
                conn,
                filename=row["filename"],
                category=row["category"],
                fold=int(row["fold"]),
            )


# ---------- Embedding ----------

def embed_corpus() -> None:
    """Load model, embed images + audio, save to a single .npz."""
    if config.INDEX_PATH.exists():
        console.log(f"[dim]index exists at {config.INDEX_PATH} — delete to re-embed[/dim]")
        return

    # Lazy import — model loading pulls in heavy deps
    from gemma_lens.model import EmbeddingModel

    model = EmbeddingModel()
    console.log(f"Loaded EmbeddingGemma 2 on {model.device} ({model.dtype})")

    with storage.connect() as conn:
        images = storage.all_images(conn)
        audios = storage.all_audio(conn)

    console.log(f"Embedding {len(images)} images + {len(audios)} audio clips")

    # Images: encode in batches with explicit PIL loading
    image_embs = _embed_images(model, images)
    audio_embs = _embed_audio(model, audios)

    config.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.INDEX_PATH,
        image_embeddings=image_embs,
        image_ids=np.array([img["id"] for img in images], dtype=np.int64),
        audio_embeddings=audio_embs,
        audio_ids=np.array([a["id"] for a in audios], dtype=np.int64),
    )
    console.log(f"[green]Saved index →[/green] {config.INDEX_PATH}")


def _embed_images(model, images: list[dict], batch: int = 32) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(images), batch), desc="embed images"):
        chunk = images[i : i + batch]
        pils = [Image.open(config.COCO_IMAGES_DIR / x["filename"]).convert("RGB") for x in chunk]
        embs = model.encode_images_batch(pils, batch_size=batch)
        all_embs.append(embs)
        for p in pils:
            p.close()
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def _embed_audio(model, audios: list[dict], batch: int = 16) -> np.ndarray:
    all_embs = []
    for i in tqdm(range(0, len(audios), batch), desc="embed audio"):
        chunk = audios[i : i + batch]
        paths = [str(config.ESC50_AUDIO_DIR / x["filename"]) for x in chunk]
        embs = model.encode_audio_paths_batch(paths, batch_size=batch)
        all_embs.append(embs)
    return np.concatenate(all_embs, axis=0).astype(np.float32)


# ---------- Entry point ----------

def main() -> None:
    console.rule("Gemma-Lens ingest")
    ensure_coco()
    ensure_esc50()
    build_metadata()
    embed_corpus()
    console.rule("[green]done[/green]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.log("[yellow]interrupted[/yellow]")
        sys.exit(130)
