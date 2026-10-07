"""In-memory search index. One .npz on disk → two torch tensors on device."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gemma_lens import config, storage


class SearchIndex:
    def __init__(self, index_path: Path | None = None, device: str | None = None):
        self.index_path = index_path or config.INDEX_PATH
        self.device = device or config.pick_device()
        data = np.load(self.index_path)
        self.image_embeddings = torch.from_numpy(data["image_embeddings"]).to(self.device)
        self.image_ids = data["image_ids"]
        self.audio_embeddings = torch.from_numpy(data["audio_embeddings"]).to(self.device)
        self.audio_ids = data["audio_ids"]

    def search(self, query: np.ndarray, k: int = config.DEFAULT_K) -> list[dict]:
        """Return top-K results across image + audio corpora, ranked by cosine similarity.

        `query` must be an L2-normalized vector of the same dim as the stored embeddings.
        """
        q = torch.from_numpy(np.asarray(query, dtype=np.float32)).to(self.device)
        if q.ndim == 2:
            q = q[0]

        scores_img = (self.image_embeddings @ q).detach().cpu().numpy()
        scores_aud = (self.audio_embeddings @ q).detach().cpu().numpy()

        hits: list[tuple[float, str, int]] = []
        hits.extend(
            (float(s), "image", int(id_)) for s, id_ in zip(scores_img, self.image_ids)
        )
        hits.extend(
            (float(s), "audio", int(id_)) for s, id_ in zip(scores_aud, self.audio_ids)
        )
        hits.sort(key=lambda x: x[0], reverse=True)
        top = hits[:k]

        # Hydrate metadata from sqlite in one pass
        results: list[dict] = []
        with storage.connect() as conn:
            for score, modality, id_ in top:
                meta = (
                    storage.get_image(conn, id_)
                    if modality == "image"
                    else storage.get_audio(conn, id_)
                )
                if meta is None:
                    continue
                results.append(
                    {
                        "id": id_,
                        "modality": modality,
                        "score": score,
                        "filename": meta["filename"],
                        "metadata": {
                            k: v for k, v in meta.items() if k not in ("id", "filename")
                        },
                        "url": _asset_url(modality, id_),
                    }
                )
        return results


def _asset_url(modality: str, id_: int) -> str:
    return f"/asset/{modality}/{id_}"
