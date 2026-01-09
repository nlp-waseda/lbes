from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from datasets import Dataset
from tqdm import tqdm
from transformers import HfArgumentParser

# =============================================================================
# Type Aliases
# =============================================================================
ChatMessage = dict[str, str]
Example = dict[str, Any]
ProblemKey = tuple[tuple[int, ...], int]

# =============================================================================
# Constants - Prompts
# =============================================================================
SYSTEM_PROMPT_TEMPLATE = """# Identity

You are a Mathematical Reasoning Expert specialized in solving arithmetic puzzles. Your task is to use ALL the given numbers EXACTLY ONCE with basic arithmetic operations to reach the target number.

# Instructions

- You MUST use each given number EXACTLY ONCE, and NO other numbers
- Use ONLY the operations: +, -, *, /
- Begin with your reasoning process inside <think></think> tags
- After the reasoning, provide your final answer inside <answer></answer> tags (e.g., <answer>{answer_example}</answer>)
- The answer content must use ONLY digits, operators, parentheses, and spaces, EXCLUDING "=" and the result"""

USER_PROMPT_TEMPLATE = """Given numbers: {numbers}
Target number: {target}"""

# =============================================================================
# Constants - Operations & Defaults
# =============================================================================
OPS: tuple[str, ...] = ("+", "-", "*", "/")

# Dataset size defaults
DEFAULT_N_TRAIN = 8192
DEFAULT_N_TEST = 1024
DEFAULT_SEED = 42

# Problem range defaults
DEFAULT_N_NUMBERS = 3
DEFAULT_NUMBERS_MIN = 1
DEFAULT_NUMBERS_MAX = 99
DEFAULT_TARGET_MIN = 1
DEFAULT_TARGET_MAX = 999

# Generation parameters
DEFAULT_MAX_DENOMINATOR = 9801
DEFAULT_MAX_EXPRESSION_ATTEMPTS = 100
DEFAULT_MAX_SAMPLE_ATTEMPTS = 1000

DEFAULT_ANSWER_EXAMPLE = "(1 + 2) / 3"


# =============================================================================
# Data Classes
# =============================================================================
@dataclass(frozen=True)
class Expr:
    value: Fraction
    text: str


@dataclass(kw_only=True)
class Config:
    # Dataset size
    n_train: int = DEFAULT_N_TRAIN
    n_test: int = DEFAULT_N_TEST
    seed: int = DEFAULT_SEED
    output_dir: str | None = None

    # Problem size
    n_given_numbers: int = DEFAULT_N_NUMBERS

    # Spec
    min_given_number: int = DEFAULT_NUMBERS_MIN
    max_given_number: int = DEFAULT_NUMBERS_MAX
    min_target_number: int = DEFAULT_TARGET_MIN
    max_target_number: int = DEFAULT_TARGET_MAX

    # Generation knobs
    max_denominator: int = DEFAULT_MAX_DENOMINATOR
    max_expression_attempts: int = DEFAULT_MAX_EXPRESSION_ATTEMPTS
    max_sample_attempts: int = DEFAULT_MAX_SAMPLE_ATTEMPTS
    answer_example: str = DEFAULT_ANSWER_EXAMPLE

    def __post_init__(self) -> None:
        self._validate_positive_fields()
        self._validate_number_bounds()
        self._validate_range_constraints()
        if not self.output_dir:
            self.output_dir = f"data/countdown{self.n_given_numbers}"

    def _validate_positive_fields(self) -> None:
        positive_fields = {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "max_denominator": self.max_denominator,
            "max_expression_attempts": self.max_expression_attempts,
            "max_sample_attempts": self.max_sample_attempts,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0")

    def _validate_number_bounds(self) -> None:
        number_bounds = {
            "min_given_number": self.min_given_number,
            "max_given_number": self.max_given_number,
            "min_target_number": self.min_target_number,
            "max_target_number": self.max_target_number,
        }
        for name, value in number_bounds.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1 (got {value})")

    def _validate_range_constraints(self) -> None:
        if self.n_given_numbers < 2:
            raise ValueError(
                f"n_given_numbers must be >= 2 (got {self.n_given_numbers})"
            )
        if self.min_given_number > self.max_given_number:
            raise ValueError("min_given_number must be <= max_given_number")
        if self.min_target_number > self.max_target_number:
            raise ValueError("min_target_number must be <= max_target_number")

    def get_generation_params(self) -> dict[str, Any]:
        return {
            "n_given_numbers": self.n_given_numbers,
            "min_given_number": self.min_given_number,
            "max_given_number": self.max_given_number,
            "min_target_number": self.min_target_number,
            "max_target_number": self.max_target_number,
            "max_denominator": self.max_denominator,
            "max_expression_attempts": self.max_expression_attempts,
            "max_sample_attempts": self.max_sample_attempts,
            "answer_example": self.answer_example,
        }


# =============================================================================
# Helper Functions
# =============================================================================
def _problem_key(numbers: Sequence[int], target: int) -> ProblemKey:
    return (tuple(sorted(numbers)), target)


def _sample_numbers(
    rng: random.Random,
    n_numbers: int,
    *,
    numbers_min: int,
    numbers_max: int,
) -> list[int]:
    return [rng.randint(numbers_min, numbers_max) for _ in range(n_numbers)]


# =============================================================================
# Expression Operations
# =============================================================================
def _apply_op(a: Expr, b: Expr, op: str) -> Expr | None:
    match op:
        case "+":
            return Expr(a.value + b.value, f"({a.text} + {b.text})")
        case "-":
            return Expr(a.value - b.value, f"({a.text} - {b.text})")
        case "*":
            return Expr(a.value * b.value, f"({a.text} * {b.text})")
        case "/":
            if b.value == 0:
                return None
            return Expr(a.value / b.value, f"({a.text} / {b.text})")
        case _:
            raise ValueError(f"Unknown op: {op}")


def _build_op_pool(
    num_examples: int, n_ops_per_example: int, rng: random.Random
) -> list[str]:
    total_ops = num_examples * n_ops_per_example
    base, remainder = divmod(total_ops, len(OPS))

    pool = [op for op in OPS for _ in range(base)]
    if remainder:
        pool.extend(rng.sample(OPS, remainder))
    rng.shuffle(pool)
    return pool


def _reduce_expressions(
    base_items: list[Expr],
    ops: Sequence[str],
    rng: random.Random,
    max_denominator: int,
) -> Expr | None:
    items = base_items[:]
    for op in ops:
        if len(items) < 2:
            return None

        i, j = rng.sample(range(len(items)), 2)
        a, b = items[i], items[j]

        if op in ("-", "/") and rng.random() < 0.5:
            a, b = b, a

        result = _apply_op(a, b, op)
        if result is None or result.value.denominator > max_denominator:
            return None

        for k in sorted((i, j), reverse=True):
            items.pop(k)
        items.append(result)

    return items[0] if len(items) == 1 else None


def _generate_target(
    given_numbers: list[int],
    ops: Sequence[str],
    rng: random.Random,
    *,
    min_target_number: int,
    max_target_number: int,
    max_denominator: int,
    max_attempts: int,
) -> int | None:
    base_items = [Expr(Fraction(n), str(n)) for n in given_numbers]

    for _ in range(max_attempts):
        expr = _reduce_expressions(base_items, ops, rng, max_denominator)
        if expr is None or expr.value.denominator != 1:
            continue

        target = int(expr.value)
        if min_target_number <= target <= max_target_number:
            return target

    return None


# =============================================================================
# Prompt Building
# =============================================================================
def _build_prompt(
    given_numbers: list[int], target: int, system_prompt: str
) -> list[ChatMessage]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        numbers=", ".join(map(str, given_numbers)),
        target=target,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =============================================================================
# Example Generation
# =============================================================================
def _try_generate_single_example(
    ops: Sequence[str],
    rng: random.Random,
    seen_keys: set[ProblemKey],
    *,
    n_given_numbers: int,
    min_given_number: int,
    max_given_number: int,
    min_target_number: int,
    max_target_number: int,
    max_denominator: int,
    max_expression_attempts: int,
    max_sample_attempts: int,
) -> tuple[list[int], int] | None:
    for _ in range(max_sample_attempts):
        candidate_numbers = _sample_numbers(
            rng,
            n_given_numbers,
            numbers_min=min_given_number,
            numbers_max=max_given_number,
        )
        candidate_target = _generate_target(
            candidate_numbers,
            ops,
            rng,
            min_target_number=min_target_number,
            max_target_number=max_target_number,
            max_denominator=max_denominator,
            max_attempts=max_expression_attempts,
        )
        if candidate_target is None:
            continue

        key = _problem_key(candidate_numbers, candidate_target)
        if key in seen_keys:
            continue

        seen_keys.add(key)
        return candidate_numbers, candidate_target

    return None


def _generate_examples(
    num_examples: int,
    rng: random.Random,
    *,
    n_given_numbers: int,
    min_given_number: int,
    max_given_number: int,
    min_target_number: int,
    max_target_number: int,
    max_denominator: int,
    max_expression_attempts: int,
    max_sample_attempts: int,
    answer_example: str,
    desc: str = "Generating examples",
) -> list[Example]:
    seen_keys = set()

    n_ops_per_example = n_given_numbers - 1
    op_pool = _build_op_pool(num_examples, n_ops_per_example, rng)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(answer_example=answer_example)

    examples: list[Example] = []
    for ex_idx in tqdm(range(num_examples), desc=desc):
        start = ex_idx * n_ops_per_example
        ops = op_pool[start : start + n_ops_per_example]

        result = _try_generate_single_example(
            ops,
            rng,
            seen_keys,
            n_given_numbers=n_given_numbers,
            min_given_number=min_given_number,
            max_given_number=max_given_number,
            min_target_number=min_target_number,
            max_target_number=max_target_number,
            max_denominator=max_denominator,
            max_expression_attempts=max_expression_attempts,
            max_sample_attempts=max_sample_attempts,
        )

        if result is None:
            raise RuntimeError(
                "Failed to generate a valid example; consider increasing attempts. "
                f"ex_idx={ex_idx}, ops={ops}"
            )

        given_numbers, target = result
        prompt = _build_prompt(given_numbers, target, system_prompt)
        examples.append({
            "prompt": prompt,
            "reward_func_args": {"target": target, "numbers": given_numbers},
        })

    return examples


# =============================================================================
# Main Entry Point
# =============================================================================
def main() -> None:
    parser = HfArgumentParser(Config)
    config: Config = parser.parse_args_into_dataclasses()[0]

    rng = random.Random(config.seed)
    gen_params = config.get_generation_params()

    train_examples = _generate_examples(
        config.n_train, rng, **gen_params, desc="Generating train"
    )
    test_examples = _generate_examples(
        config.n_test, rng, **gen_params, desc="Generating test"
    )

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Dataset.from_list(train_examples).to_parquet(str(out_dir / "train.parquet"))
    Dataset.from_list(test_examples).to_parquet(str(out_dir / "test.parquet"))


if __name__ == "__main__":
    main()
