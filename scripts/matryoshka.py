"""Matryoshka quality curve across 128/256/512/768 dims.

Measures recall@16 on a small predefined ground-truth set of text queries against
the pre-computed image corpus. Compares a truncated+re-normalized query against
truncated+re-normalized corpus vectors.

    uv run python scripts/matryoshka.py
"""

from __future__ import annotations

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from gemma_lens import config, storage
from gemma_lens.model import EmbeddingModel

console = Console()

# Predefined queries + ground-truth categories.
# Images: COCO captions contain the keywords; audio: ESC-50 category label.
# Scoring: top-16 hits must contain at least one item whose caption/category
# matches the keyword list.
GROUND_TRUTH = [
    # Visual concepts likely to live in COCO captions
    {"q": "a dog", "image_keywords": ["dog", "puppy"], "audio_categories": ["dog"]},
    {"q": "a cat", "image_keywords": ["cat", "kitten"], "audio_categories": ["cat"]},
    {"q": "a horse", "image_keywords": ["horse"], "audio_categories": []},
    {"q": "a bicycle", "image_keywords": ["bicycle", "bike"], "audio_categories": []},
    {"q": "a motorcycle", "image_keywords": ["motorcycle", "motorbike"], "audio_categories": []},
    {"q": "a red car", "image_keywords": ["red car", "red truck"], "audio_categories": []},
    {"q": "a bus", "image_keywords": ["bus"], "audio_categories": []},
    {"q": "a train", "image_keywords": ["train"], "audio_categories": ["train"]},
    {"q": "an airplane in the sky", "image_keywords": ["airplane", "plane", "jet"], "audio_categories": ["airplane"]},
    {"q": "a boat on water", "image_keywords": ["boat", "ship", "sailboat"], "audio_categories": []},
    {"q": "pizza", "image_keywords": ["pizza"], "audio_categories": []},
    {"q": "a sandwich", "image_keywords": ["sandwich"], "audio_categories": []},
    {"q": "a cake", "image_keywords": ["cake", "donut"], "audio_categories": []},
    {"q": "a laptop computer", "image_keywords": ["laptop", "computer"], "audio_categories": []},
    {"q": "a kitchen", "image_keywords": ["kitchen"], "audio_categories": []},
    {"q": "a bathroom", "image_keywords": ["bathroom", "toilet", "sink", "shower"], "audio_categories": []},
    {"q": "a bedroom", "image_keywords": ["bed", "bedroom"], "audio_categories": []},
    {"q": "a sunny beach", "image_keywords": ["beach", "sand", "surf"], "audio_categories": []},
    {"q": "snowy mountains", "image_keywords": ["snow", "mountain", "ski"], "audio_categories": []},
    {"q": "a city street", "image_keywords": ["street", "road", "traffic"], "audio_categories": []},
    {"q": "an elephant", "image_keywords": ["elephant"], "audio_categories": []},
    {"q": "a giraffe", "image_keywords": ["giraffe"], "audio_categories": []},
    {"q": "a bear in the woods", "image_keywords": ["bear"], "audio_categories": []},
    {"q": "a bird", "image_keywords": ["bird"], "audio_categories": ["chirping_birds", "crow"]},
    {"q": "a sheep", "image_keywords": ["sheep", "lamb"], "audio_categories": ["sheep"]},
    {"q": "a cow", "image_keywords": ["cow"], "audio_categories": ["cow"]},
    {"q": "a zebra", "image_keywords": ["zebra"], "audio_categories": []},
    {"q": "a person surfing", "image_keywords": ["surf", "wave"], "audio_categories": []},
    {"q": "skiing in the snow", "image_keywords": ["ski", "snow"], "audio_categories": []},
    {"q": "a baseball player", "image_keywords": ["baseball", "bat"], "audio_categories": []},
    # Sound events from ESC-50
    {"q": "a dog barking", "image_keywords": [], "audio_categories": ["dog"]},
    {"q": "a cat meowing", "image_keywords": [], "audio_categories": ["cat"]},
    {"q": "a pig oinking", "image_keywords": [], "audio_categories": ["pig"]},
    {"q": "a cow mooing", "image_keywords": [], "audio_categories": ["cow"]},
    {"q": "a frog croaking", "image_keywords": [], "audio_categories": ["frog"]},
    {"q": "a rooster crowing", "image_keywords": [], "audio_categories": ["rooster"]},
    {"q": "a hen clucking", "image_keywords": [], "audio_categories": ["hen"]},
    {"q": "crows cawing", "image_keywords": [], "audio_categories": ["crow"]},
    {"q": "birds chirping", "image_keywords": [], "audio_categories": ["chirping_birds"]},
    {"q": "crickets at night", "image_keywords": [], "audio_categories": ["crickets"]},
    {"q": "insects buzzing", "image_keywords": [], "audio_categories": ["insects"]},
    {"q": "rain falling", "image_keywords": [], "audio_categories": ["rain"]},
    {"q": "ocean waves", "image_keywords": [], "audio_categories": ["sea_waves"]},
    {"q": "a thunderstorm", "image_keywords": [], "audio_categories": ["thunderstorm"]},
    {"q": "the sound of wind", "image_keywords": [], "audio_categories": ["wind"]},
    {"q": "crackling fire", "image_keywords": [], "audio_categories": ["crackling_fire"]},
    {"q": "water drops dripping", "image_keywords": [], "audio_categories": ["water_drops", "pouring_water"]},
    {"q": "a baby crying", "image_keywords": [], "audio_categories": ["crying_baby"]},
    {"q": "a person sneezing", "image_keywords": [], "audio_categories": ["sneezing"]},
    {"q": "hands clapping", "image_keywords": [], "audio_categories": ["clapping"]},
    {"q": "heavy breathing", "image_keywords": [], "audio_categories": ["breathing"]},
    {"q": "coughing", "image_keywords": [], "audio_categories": ["coughing"]},
    {"q": "footsteps walking", "image_keywords": [], "audio_categories": ["footsteps"]},
    {"q": "people laughing", "image_keywords": [], "audio_categories": ["laughing"]},
    {"q": "someone brushing teeth", "image_keywords": [], "audio_categories": ["brushing_teeth"]},
    {"q": "snoring loudly", "image_keywords": [], "audio_categories": ["snoring"]},
    {"q": "church bells ringing", "image_keywords": [], "audio_categories": ["church_bells"]},
    {"q": "a car horn honking", "image_keywords": [], "audio_categories": ["car_horn"]},
    {"q": "a running car engine", "image_keywords": [], "audio_categories": ["engine"]},
    {"q": "an emergency siren", "image_keywords": [], "audio_categories": ["siren"]},
    {"q": "a helicopter overhead", "image_keywords": [], "audio_categories": ["helicopter"]},
    {"q": "a chainsaw cutting wood", "image_keywords": [], "audio_categories": ["chainsaw"]},
    {"q": "a train passing by", "image_keywords": [], "audio_categories": ["train"]},
    {"q": "fireworks exploding", "image_keywords": [], "audio_categories": ["fireworks"]},
    {"q": "glass shattering", "image_keywords": [], "audio_categories": ["glass_breaking"]},
    {"q": "someone typing on a keyboard", "image_keywords": [], "audio_categories": ["keyboard_typing"]},
    {"q": "a vacuum cleaner running", "image_keywords": [], "audio_categories": ["vacuum_cleaner"]},
    {"q": "a clock ticking", "image_keywords": [], "audio_categories": ["clock_tick", "clock_alarm"]},
    {"q": "a door knock", "image_keywords": [], "audio_categories": ["door_wood_knock", "door_wood_creaks"]},
    {"q": "toilet flushing", "image_keywords": [], "audio_categories": ["toilet_flush"]},
    {"q": "a washing machine", "image_keywords": [], "audio_categories": ["washing_machine"]},
]

DIMS = [128, 256, 512, 768]
K = 16


def truncate_renorm(vec: np.ndarray, dim: int) -> np.ndarray:
    v = vec[..., :dim]
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-8, None)


def recall_at_k(hits_meta: list[dict], gt: dict) -> float:
    for h in hits_meta:
        if h["modality"] == "image":
            cap = (h["caption"] or "").lower()
            if any(kw.lower() in cap for kw in gt["image_keywords"]):
                return 1.0
        else:
            cat = (h["category"] or "").lower()
            if any(c.lower() == cat for c in gt["audio_categories"]):
                return 1.0
    return 0.0


def main() -> None:
    console.rule("Matryoshka quality curve")
    # Load full 768d index + metadata
    data = np.load(config.INDEX_PATH)
    image_embs = data["image_embeddings"]
    image_ids = data["image_ids"]
    audio_embs = data["audio_embeddings"]
    audio_ids = data["audio_ids"]
    with storage.connect() as conn:
        images_meta = {i["id"]: i for i in storage.all_images(conn)}
        audios_meta = {a["id"]: a for a in storage.all_audio(conn)}

    model = EmbeddingModel()
    console.log(f"Device: {model.device} dtype: {model.dtype}")

    # Encode queries once at full dim; truncate per dim at scoring time
    query_embs = np.stack([model.encode_text(gt["q"]) for gt in GROUND_TRUTH])

    results = {}
    for dim in DIMS:
        img_t = truncate_renorm(image_embs, dim)
        aud_t = truncate_renorm(audio_embs, dim)
        q_t = truncate_renorm(query_embs, dim)

        recalls = []
        for i, gt in enumerate(GROUND_TRUTH):
            q = q_t[i]
            s_img = img_t @ q
            s_aud = aud_t @ q
            hits = []
            for score, id_ in zip(s_img, image_ids):
                hits.append({"score": float(score), "modality": "image",
                             "caption": images_meta.get(int(id_), {}).get("caption"),
                             "category": None})
            for score, id_ in zip(s_aud, audio_ids):
                hits.append({"score": float(score), "modality": "audio",
                             "caption": None,
                             "category": audios_meta.get(int(id_), {}).get("category")})
            hits.sort(key=lambda h: h["score"], reverse=True)
            recalls.append(recall_at_k(hits[:K], gt))
        results[dim] = sum(recalls) / len(recalls)

    table = Table(title=f"recall@{K} on {len(GROUND_TRUTH)} known-good queries")
    table.add_column("dim", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("vs 768d", justify="right")
    base = results[768]
    for dim in DIMS:
        rel = f"{results[dim] / base * 100:.0f}%" if base > 0 else "n/a"
        table.add_row(str(dim), f"{results[dim]:.3f}", rel)
    console.print(table)


if __name__ == "__main__":
    main()
