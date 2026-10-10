"""GRPO fine-tuning of a VLM on counting / spatial questions with verifiable rewards.

    accelerate launch train_grpo.py --config configs/qwen2_5_vl_3b_lora.yaml

Any GRPOConfig / ModelConfig field can be overridden on the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from datasets import Dataset, Image, Sequence
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config

from spatial_r1.data import build_examples
from spatial_r1.rewards import accuracy_reward, format_reward, overlong_penalty


@dataclass
class DataArguments:
    data_dir: str = field(default="data/ssb", metadata={"help": "spatial-scene-bench output directory"})
    modality: str | None = field(default=None, metadata={"help": "image | video | None for both"})
    types: list[str] | None = field(default=None)
    max_frames: int = 8
    val_fraction: float = 0.1
    dense_counts: bool = True
    max_think_words: int = 300
    reward_weights_override: list[float] | None = field(default=None, metadata={"help": "format, accuracy, length"})


def to_dataset(examples: list[dict]) -> Dataset:
    ds = Dataset.from_list(examples).rename_column("image_paths", "images")
    return ds.cast_column("images", Sequence(Image()))


def main() -> None:
    parser = TrlParser((DataArguments, GRPOConfig, ModelConfig))
    data_args, train_args, model_args = parser.parse_args_and_config()

    train, val = build_examples(data_args.data_dir, data_args.modality, data_args.types, data_args.max_frames,
                                val_fraction=data_args.val_fraction, seed=train_args.seed)
    acc = partial(accuracy_reward, dense_counts=data_args.dense_counts)
    acc.__name__ = "accuracy_reward"
    length = partial(overlong_penalty, max_think_words=data_args.max_think_words)
    length.__name__ = "overlong_penalty"
    if data_args.reward_weights_override:
        train_args.reward_weights = data_args.reward_weights_override
    elif train_args.reward_weights is None:
        train_args.reward_weights = [0.5, 1.0, 0.2]
    train_args.model_init_kwargs = {
        **(train_args.model_init_kwargs or {}),
        "dtype": model_args.dtype,
        "attn_implementation": model_args.attn_implementation,
    }

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward, acc, length],
        args=train_args,
        train_dataset=to_dataset(train),
        eval_dataset=to_dataset(val) if val and train_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    trainer.save_model(train_args.output_dir)


if __name__ == "__main__":
    main()
