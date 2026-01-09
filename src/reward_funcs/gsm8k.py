import re

_PATTERN = re.compile(
    r"^<think>.+</think>.*<answer>(-?[\d,]*\d)</answer>.*$", re.DOTALL
)


def _extract_answer(text: str) -> str | None:
    match = _PATTERN.match(text)
    if (
        text.count("<think>") == 1
        and text.count("</think>") == 1
        and text.count("<answer>") == 1
        and text.count("</answer>") == 1
        and match is not None
    ):
        return match.group(1)
    return None


# "reward" is necessary
def compute_score(completion_text: str, ground_truth: str) -> dict[str, float]:
    answer = _extract_answer(completion_text)
    if answer is None:
        return {"reward": 0.0, "format": 0.0, "accuracy": 0.0}
    else:
        if answer.replace(",", "") == ground_truth.replace(",", ""):
            return {"reward": 1.1, "format": 0.1, "accuracy": 1.0}
        else:
            return {"reward": 0.1, "format": 0.1, "accuracy": 0.0}
