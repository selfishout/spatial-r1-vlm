"""Turn spatial-scene-bench questions into GRPO / SFT training examples."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

R1_INSTRUCTION = (
    "First think about the 3D layout of the scene step by step inside <think></think> tags "
    "(identify the relevant objects, where they are, and check each one), then give only the final "
    "answer inside <answer></answer> tags."
)

ANSWER_HINT = {
    "count": "The final answer is a single integer.",
    "yesno": "The final answer is yes or no.",
    "choice": "The final answer is the letter of the correct option.",
    "numeric": "The final answer is a number in meters.",
}


def load_questions(data_dir: str | Path) -> list[dict]:
    with open(Path(data_dir) / "questions.jsonl") as f:
        return [json.loads(line) for line in f]


def split_by_scene(records: list[dict], val_fraction: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Scene-disjoint split so that validation scenes are never seen during training."""
    scenes = sorted({r["scene_id"] for r in records})
    random.Random(seed).shuffle(scenes)
    val = set(scenes[: max(1, int(len(scenes) * val_fraction))])
    return [r for r in records if r["scene_id"] not in val], [r for r in records if r["scene_id"] in val]


def balance_types(records: list[dict], per_type: int | None = None, seed: int = 0) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    rng = random.Random(seed)
    cap = per_type or min(len(v) for v in by_type.values())
    out = []
    for v in by_type.values():
        rng.shuffle(v)
        out.extend(v[:cap])
    rng.shuffle(out)
    return out


def subsample_frames(paths: list[str], max_frames: int) -> list[str]:
    if len(paths) <= max_frames:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
    return [paths[i] for i in idx]


def to_example(rec: dict, data_dir: str | Path, max_frames: int = 8) -> dict:
    """Conversational GRPO example with image paths; TRL loads images from the ``images`` column."""
    paths = [str(Path(data_dir) / p) for p in subsample_frames(rec["images"], max_frames)]
    text = f"{rec['question']}\n{R1_INSTRUCTION} {ANSWER_HINT[rec['answer_type']]}"
    content = [{"type": "image"} for _ in paths] + [{"type": "text", "text": text}]
    return {
        "prompt": [{"role": "user", "content": content}],
        "image_paths": paths,
        "solution": rec["answer"],
        "answer_type": rec["answer_type"],
        "options": rec.get("options") or [],
        "type": rec["type"],
        "id": rec["id"],
    }


def build_examples(data_dir: str | Path, modality: str | None = None, types: list[str] | None = None,
                   max_frames: int = 8, balance: bool = True, val_fraction: float = 0.1,
                   seed: int = 0) -> tuple[list[dict], list[dict]]:
    recs = load_questions(data_dir)
    if modality:
        recs = [r for r in recs if r["modality"] == modality]
    if types:
        recs = [r for r in recs if r["type"] in types]
    train, val = split_by_scene(recs, val_fraction, seed)
    if balance:
        train = balance_types(train, seed=seed)
    return ([to_example(r, data_dir, max_frames) for r in train],
            [to_example(r, data_dir, max_frames) for r in val])
