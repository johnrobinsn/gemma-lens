"""Benchmark TTFE per modality + end-to-end /search latency.

    uv run python scripts/bench.py
"""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

from gemma_lens import config, storage
from gemma_lens.index import SearchIndex
from gemma_lens.model import EmbeddingModel

console = Console()
RNG = random.Random(0)

TEXT_QUERIES = [
    "red cars", "sunset over water", "snowy mountains", "kitchen with food",
    "dogs playing in grass", "white objects", "person riding a bicycle",
    "crowded city street", "animals in the wild", "food on a plate",
]


def _stats(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "n": len(s),
        "median_ms": statistics.median(s) * 1000,
        "p99_ms": s[int(0.99 * (len(s) - 1))] * 1000,
        "mean_ms": statistics.mean(s) * 1000,
    }


def bench_text(model: EmbeddingModel, n: int = 100) -> dict:
    times = []
    for _ in range(n):
        q = RNG.choice(TEXT_QUERIES)
        t0 = time.perf_counter()
        model.encode_text(q)
        times.append(time.perf_counter() - t0)
    return _stats(times)


def bench_image(model: EmbeddingModel, images: list[dict], n: int = 100) -> dict:
    times = []
    sample = [RNG.choice(images) for _ in range(n)]
    pils = [Image.open(config.COCO_IMAGES_DIR / x["filename"]).convert("RGB") for x in sample]
    for pil in pils:
        t0 = time.perf_counter()
        model.encode_image(pil)
        times.append(time.perf_counter() - t0)
    return _stats(times)


def bench_audio(model: EmbeddingModel, audios: list[dict], n: int = 100) -> dict:
    import librosa

    times = []
    sample = [RNG.choice(audios) for _ in range(n)]
    waves = []
    for a in sample:
        wav, sr = librosa.load(config.ESC50_AUDIO_DIR / a["filename"], sr=16000, mono=True)
        waves.append((np.asarray(wav, dtype=np.float32), 16000))
    for wav, sr in waves:
        t0 = time.perf_counter()
        model.encode_audio(wav, sr)
        times.append(time.perf_counter() - t0)
    return _stats(times)


def bench_end_to_end(model: EmbeddingModel, index: SearchIndex, n: int = 100) -> dict:
    times_total, times_embed, times_search = [], [], []
    for _ in range(n):
        q = RNG.choice(TEXT_QUERIES)
        t0 = time.perf_counter()
        emb = model.encode_text(q)
        t1 = time.perf_counter()
        index.search(emb, k=16)
        t2 = time.perf_counter()
        times_total.append(t2 - t0)
        times_embed.append(t1 - t0)
        times_search.append(t2 - t1)
    return {
        "total": _stats(times_total),
        "embed": _stats(times_embed),
        "search": _stats(times_search),
    }


def main() -> None:
    console.rule("Gemma-Lens benchmark")
    model = EmbeddingModel()
    index = SearchIndex()
    with storage.connect() as conn:
        images = storage.all_images(conn)
        audios = storage.all_audio(conn)

    console.log(f"Device: {model.device} dtype: {model.dtype}")
    console.log(f"Corpus: {len(images)} images, {len(audios)} audio")

    # Warmup
    for _ in range(5):
        model.encode_text("warmup")

    text = bench_text(model)
    image = bench_image(model, images)
    audio = bench_audio(model, audios)
    e2e = bench_end_to_end(model, index)

    table = Table(title="TTFE (time-to-first-embedding)")
    table.add_column("modality"); table.add_column("n")
    table.add_column("median (ms)", justify="right"); table.add_column("p99 (ms)", justify="right")
    for name, s in [("text", text), ("image", image), ("audio", audio)]:
        table.add_row(name, str(s["n"]), f"{s['median_ms']:.1f}", f"{s['p99_ms']:.1f}")
    console.print(table)

    e2etable = Table(title="End-to-end /search (text queries)")
    e2etable.add_column("stage"); e2etable.add_column("n")
    e2etable.add_column("median (ms)", justify="right"); e2etable.add_column("p99 (ms)", justify="right")
    for name in ("total", "embed", "search"):
        s = e2e[name]
        e2etable.add_row(name, str(s["n"]), f"{s['median_ms']:.1f}", f"{s['p99_ms']:.1f}")
    console.print(e2etable)


if __name__ == "__main__":
    main()
