# Gemma-Lens

Multimodal search engine demo for Google [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (740M, released 2026-10-06). Query a mixed corpus of ~5,000 COCO images + 2,000 ESC-50 audio clips with **text**, **voice**, or an **uploaded image** — results are ranked by cosine similarity in a single 768-d space.

Companion blog article (publishing 2026-10-14): **[Multimodal search in 200 lines — EmbeddingGemma 2, direct cosine, zero FAISS](https://www.storminthecastle.com/posts/gemma_lens/)**

## Quickstart

```bash
# 1. Clone
git clone https://github.com/johnrobinsn/gemma-lens.git
cd gemma-lens

# 2. Install (uv handles everything)
uv sync

# 3. Download data + build embeddings (one-time, ~25 min on RTX 5090)
uv run python -m gemma_lens.ingest

# 4. Run
uv run gemma-lens
# → serves on http://localhost:8000
```

### Prerequisites

- NVIDIA GPU with ≥ 4 GB VRAM (CPU-only works too — see below)
- CUDA 12.x
- Python 3.12+
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- ~3 GB free disk
- `ffmpeg` on PATH (required for browser-recorded webm/opus audio decoding)

### Picking a device

Gemma-Lens auto-detects GPU on startup. Override with standard CUDA env vars:

```bash
# Use a specific GPU (e.g. second card in nvidia-smi order)
CUDA_VISIBLE_DEVICES=1 uv run gemma-lens
CUDA_VISIBLE_DEVICES=1 uv run python -m gemma_lens.ingest

# Force CPU (works on any machine, no CUDA needed)
CUDA_VISIBLE_DEVICES="" uv run gemma-lens
```

**CPU performance** on a modern multi-core host is surprisingly usable for a demo: ~60–90 ms per text query (vs ~50 ms on an RTX 5090). Image and audio queries are substantially slower on CPU because their encoders are larger — expect 1–3 seconds per query. Ingest on CPU will take hours instead of minutes; if you can borrow a GPU just for the ingest step and then run the server on CPU, that's a reasonable split.

## How it works

- **Embed**: EmbeddingGemma 2 projects text, images, and audio into a shared 768-d space.
- **Index**: Vectors are L2-normalized and stored as a single `.npz` on disk; loaded as a torch tensor on startup.
- **Retrieve**: Each query becomes a 768-d vector; one matmul against the corpus matrix returns top-K matches. No FAISS needed at this scale (~7,000 items × 768d ≈ 22 MB).
- **Display**: Single-page UI shows ranked images + audio tiles with similarity scores.

## Benchmarks

```bash
uv run python scripts/bench.py       # TTFE per modality + end-to-end latency
uv run python scripts/matryoshka.py  # Quality curve at 128/256/512/768 dims
```

## Try your own corpus

Replace the `data/images/val2017/*.jpg` and `data/esc50/audio/*.wav` directories with your own files, delete `data/index.npz`, and re-run `uv run python -m gemma_lens.ingest`.

## Licenses

- **Code**: [MIT](LICENSE).
- **EmbeddingGemma 2 weights**: Apache 2.0 (Google).
- **COCO 2017 dataset**: CC BY 4.0.
- **ESC-50 dataset**: **CC BY-NC 4.0** — non-commercial research use only. The four preset clips shipped in `static/presets/` carry individual Freesound creator attributions — see [static/presets/ATTRIBUTIONS.md](static/presets/ATTRIBUTIONS.md).

## Links

- EmbeddingGemma 2 model card: https://huggingface.co/google/embeddinggemma-2
- Google launch post: https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/
- COCO 2017: https://cocodataset.org
- ESC-50: https://github.com/karolpiczak/ESC-50
