"""Verifiable rewards for RL on counting and spatial questions.

All functions follow the TRL reward signature ``f(completions, **dataset_columns) -> list[float]``
and accept both plain-string and conversational completions.
"""

from __future__ import annotations

import re

from spatial_scene_bench.evaluate import mean_relative_accuracy, parse_answer

FORMAT_RE = re.compile(r"^\s*<think>(.+?)</think>\s*<answer>(.+?)</answer>\s*$", re.S)


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        content = completion[-1].get("content", "")
        if isinstance(content, list):
            return "".join(c.get("text", "") for c in content if isinstance(c, dict))
        return content
    return str(completion)


def format_reward(completions, **kwargs) -> list[float]:
    """1.0 iff the completion is exactly ``<think>...</think><answer>...</answer>``."""
    return [1.0 if FORMAT_RE.match(_text(c)) else 0.0 for c in completions]


def _answer_span(text: str) -> str | None:
    m = FORMAT_RE.match(text)
    return m.group(2) if m else None


def accuracy_reward(completions, solution, answer_type, options=None, dense_counts: bool = True,
                    **kwargs) -> list[float]:
    """Task reward computed only from the <answer> span.

    * yes/no and multiple choice: exact match
    * numeric (metric distance): VSI-Bench mean relative accuracy in [0, 1]
    * counts: exact match, or with ``dense_counts`` 0.5 * MRA partial credit for
      near-misses so that early policies still get a learning signal
    """
    rewards = []
    options = options or [None] * len(completions)
    for c, sol, at, opts in zip(completions, solution, answer_type, options):
        span = _answer_span(_text(c))
        if span is None:
            rewards.append(0.0)
            continue
        pred = parse_answer(f"<answer>{span}</answer>", at, opts)
        if pred is None:
            rewards.append(0.0)
        elif at == "numeric":
            rewards.append(mean_relative_accuracy(float(pred), float(sol)))
        elif at == "count":
            if int(pred) == int(sol):
                rewards.append(1.0)
            else:
                rewards.append(0.5 * mean_relative_accuracy(float(pred), float(sol)) if dense_counts else 0.0)
        else:
            rewards.append(1.0 if pred == sol else 0.0)
    return rewards


def overlong_penalty(completions, max_think_words: int = 300, **kwargs) -> list[float]:
    """Soft penalty for rambling reasoning chains (0 within budget, down to -1 at 2x budget)."""
    out = []
    for c in completions:
        m = FORMAT_RE.match(_text(c))
        n = len(m.group(1).split()) if m else 0
        out.append(-min(1.0, max(0.0, (n - max_think_words) / max_think_words)))
    return out


REWARD_FUNCS = {"format": format_reward, "accuracy": accuracy_reward, "length": overlong_penalty}
