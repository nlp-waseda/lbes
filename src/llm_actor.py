import dataclasses

import numpy as np
import ray
from transformers import AutoConfig, GenerationConfig
from vllm import LLM, EngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.model_loader import ShardedStateLoader


@ray.remote
class LLMActor:
    def __init__(self, engine_args: EngineArgs) -> None:
        self.engine_args = dataclasses.replace(engine_args)
        self.engine_args.worker_extension_cls = "worker_extension.WorkerExtension"
        self.llm = LLM(**dataclasses.asdict(self.engine_args))
        self.tokenizer = self.llm.get_tokenizer()

    def ready(self) -> bool:
        return True

    def add_noise(self, seed: int, noise_std: float) -> None:
        self.llm.collective_rpc("add_noise", args=(seed, noise_std))
        self.llm.reset_prefix_cache()

    def remove_noise(self, seed: int, noise_std: float) -> None:
        self.llm.collective_rpc("remove_noise", args=(seed, noise_std))
        self.llm.reset_prefix_cache()

    def update(
        self, seeds: list[int], advantages: list[float], lr: float, noise_std: float
    ) -> None:
        self.llm.collective_rpc("update", args=(seeds, advantages, lr, noise_std))
        self.llm.reset_prefix_cache()

    def generate(
        self, prompts: list[list[dict[str, str]]], sampling_params: SamplingParams
    ) -> dict[
        str,
        list[list[str]] | list[list[int]] | list[list[list[int]]] | list[list[bool]],
    ]:
        outputs = self.llm.chat(prompts, sampling_params, use_tqdm=False)
        completion_texts = [
            [choice.text for choice in output.outputs] for output in outputs
        ]
        prompt_token_ids = [output.prompt_token_ids for output in outputs]
        completion_token_ids = [
            [list(choice.token_ids) for choice in output.outputs] for output in outputs
        ]
        is_terminated = [
            [choice.finish_reason != "length" for choice in output.outputs]
            for output in outputs
        ]
        return {
            "completion_texts": completion_texts,
            "prompt_token_ids": prompt_token_ids,
            "completion_token_ids": completion_token_ids,
            "is_terminated": is_terminated,
        }

    def calc_mean_logprobs(
        self,
        prompt_token_ids: list[list[int]],  # (batch_size, )
        completion_token_ids: list[list[list[int]]],  # (batch_size, n)
    ) -> list[list[float]]:  # (batch_size, n)
        tokens_prompts = []
        for sample_prompt_token_ids, sample_completion_token_ids in zip(
            prompt_token_ids, completion_token_ids
        ):
            for choice_completion_token_ids in sample_completion_token_ids:
                tokens_prompt = TokensPrompt(
                    prompt_token_ids=sample_prompt_token_ids
                    + choice_completion_token_ids
                )
                tokens_prompts.append(tokens_prompt)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            flat_logprobs=True,
            detokenize=False,
        )

        outputs = self.llm.generate(tokens_prompts, sampling_params, use_tqdm=False)

        mean_logprobs = []
        pos = 0
        for i in range(len(prompt_token_ids)):
            n = len(completion_token_ids[i])
            sample_outputs = outputs[pos : pos + n]
            prompt_length = len(prompt_token_ids[i])
            sample_mean_logprobs = []
            for choice_output in sample_outputs:
                response_logprobs = [
                    choice_output.prompt_logprobs.logprobs[start_idx]
                    for start_idx in choice_output.prompt_logprobs.start_indices[
                        prompt_length:
                    ]
                ]
                mean_logprob = np.mean(response_logprobs).item()
                sample_mean_logprobs.append(mean_logprob)
            mean_logprobs.append(sample_mean_logprobs)
            pos += n

        # (batch_size, n)
        return mean_logprobs

    def calc_entropy(
        self,
        prompt_token_ids: list[list[int]],  # (batch_size, )
        completion_token_ids: list[list[list[int]]],  # (batch_size, n)
        max_logprobs: int,
    ) -> list[list[float]]:  # (batch_size, n)
        tokens_prompts = []
        for sample_prompt_token_ids, sample_completion_token_ids in zip(
            prompt_token_ids, completion_token_ids
        ):
            for choice_completion_token_ids in sample_completion_token_ids:
                tokens_prompt = TokensPrompt(
                    prompt_token_ids=sample_prompt_token_ids
                    + choice_completion_token_ids
                )
                tokens_prompts.append(tokens_prompt)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=max_logprobs,
            flat_logprobs=True,
            detokenize=False,
        )

        outputs = self.llm.generate(tokens_prompts, sampling_params, use_tqdm=False)

        batch_entropy = []
        pos = 0
        for i in range(len(prompt_token_ids)):
            n = len(completion_token_ids[i])
            sample_outputs = outputs[pos : pos + n]
            prompt_length = len(prompt_token_ids[i])
            sample_entropy = []
            for choice_output in sample_outputs:
                entropy = []
                for start_idx, end_idx in zip(
                    choice_output.prompt_logprobs.start_indices[prompt_length:],
                    choice_output.prompt_logprobs.end_indices[prompt_length:],
                ):
                    logprobs = np.array(
                        choice_output.prompt_logprobs.logprobs[start_idx:end_idx]
                    )
                    probs = np.exp(logprobs)
                    entropy.append(-np.sum(probs * logprobs) / np.log(len(probs)))
                choice_entropy = np.mean(entropy)
                sample_entropy.append(choice_entropy.item())
            batch_entropy.append(sample_entropy)
            pos += n

        # (batch_size, n)
        return batch_entropy

    def save_model(self, output_dir: str) -> None:
        if self.engine_args.tensor_parallel_size == 1:
            self.llm.collective_rpc("save_safetensors", args=(output_dir,))
        else:
            self.llm.llm_engine.engine_core.save_sharded_state(
                path=output_dir,
                pattern=ShardedStateLoader.DEFAULT_PATTERN,
                max_size=5 * 1024**3,
            )

        cfg = AutoConfig.from_pretrained(
            self.engine_args.model,
            trust_remote_code=self.engine_args.trust_remote_code,
        )
        cfg.save_pretrained(output_dir)

        try:
            gen_cfg = GenerationConfig.from_pretrained(
                self.engine_args.model,
                trust_remote_code=self.engine_args.trust_remote_code,
            )
            gen_cfg.save_pretrained(output_dir)
        except Exception:
            pass

        self.llm.get_tokenizer().save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")
