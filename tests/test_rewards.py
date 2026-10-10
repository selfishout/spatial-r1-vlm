import pytest

from spatial_r1.rewards import accuracy_reward, format_reward, overlong_penalty


def conv(text):
    return [{"role": "assistant", "content": text}]


GOOD = "<think>I see two red cubes and one more behind.</think><answer>3</answer>"


def test_format_reward_string_and_conversational():
    assert format_reward([GOOD, conv(GOOD), "3", "<answer>3</answer>"]) == [1.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("completion,sol,at,expected", [
    (GOOD, "3", "count", 1.0),
    ("<think>x</think><answer>yes</answer>", "yes", "yesno", 1.0),
    ("<think>x</think><answer>no</answer>", "yes", "yesno", 0.0),
    ("<think>x</think><answer>(B) front-right</answer>", "B", "choice", 1.0),
    ("<think>x</think><answer>2.0</answer>", "2.00", "numeric", 1.0),
    ("<think>x</think><answer>10</answer>", "2.00", "numeric", 0.0),
    ("I think 3", "3", "count", 0.0),  # right number but wrong format earns nothing
])
def test_accuracy_reward(completion, sol, at, expected):
    assert accuracy_reward([completion], [sol], [at], [["front-left", "front-right", "back-left", "back-right"]]) == [
        pytest.approx(expected)]


def test_dense_count_partial_credit():
    near = "<think>x</think><answer>4</answer>"
    dense = accuracy_reward([near], ["5"], ["count"])[0]
    sparse = accuracy_reward([near], ["5"], ["count"], dense_counts=False)[0]
    assert 0 < dense < 0.5 and sparse == 0.0


def test_overlong_penalty():
    short = "<think>" + "w " * 10 + "</think><answer>1</answer>"
    long = "<think>" + "w " * 600 + "</think><answer>1</answer>"
    assert overlong_penalty([short, long], max_think_words=300) == [0.0, -1.0]
