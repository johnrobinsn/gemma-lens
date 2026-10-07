"""EmbeddingGemma 2 wrapper via sentence-transformers.

Public surface is three per-modality encode methods. Internally they all call
`SentenceTransformer.encode(...)` with the right input shape.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import PIL.Image
import torch
from sentence_transformers import SentenceTransformer

from gemma_lens import config

PathLike = Union[str, "PIL.Image.Image"]


class EmbeddingModel:
    def __init__(
        self,
        device: str | None = None,
        dtype: str = "auto",
        truncate_dim: int | None = None,
    ):
        self.device = device or config.pick_device()
        self.dtype = config.pick_dtype(self.device, dtype)
        self.truncate_dim = truncate_dim
        self.model = SentenceTransformer(
            config.MODEL_ID,
            device=self.device,
            model_kwargs={"torch_dtype": self.dtype},
        )

    def _encode_kwargs(self, extra: dict | None = None) -> dict:
        kwargs = {"normalize_embeddings": True, "convert_to_numpy": True}
        if self.truncate_dim:
            kwargs["truncate_dim"] = self.truncate_dim
        if extra:
            kwargs.update(extra)
        return kwargs

    def encode_text(self, text: str, task: str = "SearchQuery") -> np.ndarray:
        emb = self.model.encode(text, prompt_name=task, **self._encode_kwargs())
        return np.asarray(emb, dtype=np.float32)

    def encode_image(self, image: PIL.Image.Image) -> np.ndarray:
        emb = self.model.encode({"image": image}, **self._encode_kwargs())
        return np.asarray(emb, dtype=np.float32)

    def encode_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        wav = _resample_to_16k_mono(waveform, sr)
        emb = self.model.encode({"audio": wav}, **self._encode_kwargs())
        return np.asarray(emb, dtype=np.float32)

    # Batched variants used by the ingest script. The model accepts a list of
    # dicts for multimodal batches.

    def encode_images_batch(
        self, images: list[PIL.Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        payload = [{"image": img} for img in images]
        emb = self.model.encode(
            payload, batch_size=batch_size, show_progress_bar=False,
            **self._encode_kwargs()
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_audio_paths_batch(
        self, paths: list[str], batch_size: int = 16
    ) -> np.ndarray:
        """Paths-only batch: sentence-transformers decodes + resamples internally."""
        payload = [{"audio": p} for p in paths]
        emb = self.model.encode(
            payload, batch_size=batch_size, show_progress_bar=False,
            **self._encode_kwargs()
        )
        return np.asarray(emb, dtype=np.float32)


def _resample_to_16k_mono(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Flatten to mono, resample to 16 kHz. librosa handles both."""
    import librosa

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0 if waveform.shape[0] < waveform.shape[-1] else -1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if sr != config.AUDIO_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=config.AUDIO_SR)
    return waveform
