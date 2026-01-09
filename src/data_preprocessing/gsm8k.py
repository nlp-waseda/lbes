from datasets import load_dataset

SYSTEM_PROMPT = """You are a reasoning model designed to think step-by-step before providing answers. Always follow this format:

1. Start your response with your thinking process inside `<think></think>` tags
2. After your thinking, provide your explanation and include the final numerical answer inside `<answer></answer>` tags

In your `<think>` section:
- Break down the problem into smaller parts
- Consider different approaches or perspectives
- Work through your reasoning step by step
- Identify any assumptions you're making
- Check your logic for potential errors

Your thinking should be thorough but concise. After closing the `</think>` tag, provide your explanation and wrap the numerical answer within `<answer></answer>` tags. The answer tags should contain only the number."""


def process(example):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    ground_truth = example["answer"].split("####")[-1].strip()
    return {"prompt": prompt, "reward_func_args": {"ground_truth": ground_truth}}


def main():
    dataset = load_dataset("openai/gsm8k", "main")

    train_dataset = dataset["train"].map(
        process, remove_columns=dataset["train"].column_names
    )
    test_dataset = dataset["test"].map(
        process, remove_columns=dataset["test"].column_names
    )

    train_dataset.to_parquet("data/gsm8k/train.parquet")
    test_dataset.to_parquet("data/gsm8k/test.parquet")


if __name__ == "__main__":
    main()
