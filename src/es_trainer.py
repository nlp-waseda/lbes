import dataclasses
import importlib
import os
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
import wandb
from datasets import Dataset, load_dataset
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import EngineArgs, SamplingParams

from es_config import ESConfig
from llm_actor import LLMActor
from schedulers import create_scheduler
from seed_utils import derive_seed, set_seed

ADVANTAGE_STD_EPSILON = 1e-6


# TSUBAME4.0 specific patch to avoid Ray initialization error
def _patched_get_system_processes_for_resource_isolation(self):
    return ""


ray._private.node.Node._get_system_processes_for_resource_isolation = (
    _patched_get_system_processes_for_resource_isolation
)


class ESTrainer:
    """Evolution Strategies Trainer"""

    @staticmethod
    def simple_collater(batch: Dataset) -> dict[str, list[Any]]:
        return {k: [d[k] for d in batch] for k in batch[0].keys()}

    def __init__(self, config: ESConfig) -> None:
        self.config = config
        self.setup()

    def setup(self) -> None:
        set_seed(self.config.seed)

        self.engine_args = EngineArgs(
            model=self.config.model,
            load_format=self.config.load_format,
            max_model_len=self.config.max_model_len,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
        )

        self.sampling_params = SamplingParams(
            n=self.config.n,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        train_dataset = load_dataset(
            "parquet", data_files={"train": self.config.train_file}, split="train"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data_batch_size,
            shuffle=True,
            collate_fn=self.simple_collater,
        )
        if self.config.eval_file is not None:
            self.eval_dataset = load_dataset(
                "parquet", data_files={"eval": self.config.eval_file}, split="eval"
            )
        self.reward_func = self.load_reward_func(self.config.reward_func_path)
        self.total_steps = len(self.train_loader) * self.config.num_train_epochs

        self.noise_std_scheduler = create_scheduler(
            self.config.noise_std_scheduler_type,
            self.config.noise_std,
            self.total_steps,
            final_value=self.config.noise_std_final,
        )
        self.lr_scheduler = create_scheduler(
            self.config.lr_scheduler_type,
            self.config.learning_rate,
            self.total_steps,
            final_value=self.config.lr_final,
        )

        if self.config.report_to == "wandb":
            wandb.init(
                name=self.config.run_name, config=dataclasses.asdict(self.config)
            )
        self.global_step = 0
        self.train_num_tokens = 0
        self.saved_dirs = []
        self.train_time_total = 0.0

        self.initialize_actors()

    @staticmethod
    def load_reward_func(reward_func_path: str) -> Callable[..., dict[str, float]]:
        sys.path.insert(0, os.getcwd())

        module_path, func_name = reward_func_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Failed to import module '{module_path}' "
                f"for reward function '{reward_func_path}'"
            ) from exc
        try:
            reward_func = getattr(module, func_name)
        except AttributeError as exc:
            raise ImportError(
                f"Reward function '{func_name}' not found in module '{module_path}'"
            ) from exc
        return reward_func

    def initialize_actors(self) -> None:
        num_cpus = int(os.environ.get("NSLOTS", 1))
        num_gpus = torch.cuda.device_count()

        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

        if num_gpus < self.engine_args.tensor_parallel_size:
            raise ValueError(
                f"Need more than {self.engine_args.tensor_parallel_size} GPUs, "
                f"but only {num_gpus} available"
            )

        num_actors = min(
            self.config.population_size,
            num_gpus // self.engine_args.tensor_parallel_size,
        )

        bundles = [
            {
                "GPU": self.engine_args.tensor_parallel_size,
                "CPU": 0,
            }
            for _ in range(num_actors)
        ]
        self.pg = placement_group(bundles)
        ray.get(self.pg.ready())

        self.actors = [
            LLMActor.options(
                num_cpus=0,
                num_gpus=self.engine_args.tensor_parallel_size,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg,
                    placement_group_bundle_index=i,
                    placement_group_capture_child_tasks=True,
                ),
            ).remote(self.engine_args)
            for i in range(num_actors)
        ]

        ray.get([actor.ready.remote() for actor in self.actors])

        print(f"Initialized {len(self.actors)} actors")

    def generate(
        self, prompts: list[list[dict[str, str]]], sampling_params: SamplingParams
    ) -> dict[
        str,
        list[list[str]] | list[list[int]] | list[list[list[int]]] | list[list[bool]],
    ]:
        num_actors = len(self.actors)
        q, m = divmod(len(prompts), num_actors)
        split_prompts = [
            prompts[i * q + min(i, m) : (i + 1) * q + min(i + 1, m)]
            for i in range(num_actors)
        ]
        split_prompts = [sub_prompts for sub_prompts in split_prompts if sub_prompts]

        results = ray.get([
            actor.generate.remote(sub_prompts, sampling_params)
            for actor, sub_prompts in zip(self.actors, split_prompts)
        ])
        merged_results = {
            k: list(chain.from_iterable(d[k] for d in results))
            for k in results[0].keys()
        }

        return merged_results

    def generate_with_noise(
        self,
        prompts: list[list[dict[str, str]]],
        sampling_params: SamplingParams,
        seeds: list[int],
        current_noise_std: float,
    ) -> dict[
        str,
        list[list[list[str]]]
        | list[list[list[int]]]
        | list[list[list[list[int]]]]
        | list[list[list[bool]]],
    ]:
        num_actors = len(self.actors)
        num_seeds = len(seeds)
        all_results = []
        pos = 0
        while pos < num_seeds:
            num_noises = min(num_actors, num_seeds - pos)
            noise_actors = self.actors[:num_noises]

            ray.get([
                actor.add_noise.remote(seed, current_noise_std)
                for actor, seed in zip(noise_actors, seeds[pos : pos + num_noises])
            ])

            results = ray.get([
                actor.generate.remote(prompts, sampling_params)
                for actor in noise_actors
            ])
            all_results.extend(results)

            ray.get([
                actor.remove_noise.remote(seed, current_noise_std)
                for actor, seed in zip(noise_actors, seeds[pos : pos + num_noises])
            ])

            pos += num_noises

        merged_results = {k: [d[k] for d in all_results] for k in all_results[0].keys()}

        return merged_results

    def warmup_generate(self) -> None:
        prompts = [[{"role": "user", "content": "Warmup"}]]
        sampling_params = self.sampling_params.clone()
        sampling_params.max_tokens = 1
        ray.get([
            actor.generate.remote(prompts, sampling_params) for actor in self.actors
        ])

    def update_all_actors(
        self, seeds: list[int], advantages: list[float], lr: float, noise_std: float
    ) -> None:
        ray.get([
            actor.update.remote(seeds, advantages, lr, noise_std)
            for actor in self.actors
        ])

    def save_checkpoint(self, step: int) -> None:
        if len(self.saved_dirs) >= self.config.save_total_limit:
            oldest = self.saved_dirs.pop(0)
            if oldest.exists():
                shutil.rmtree(oldest)

        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint at step {step}...")
        ray.get(self.actors[0].save_model.remote(checkpoint_dir))

        self.saved_dirs.append(checkpoint_dir)

    def save_model(self, output_dir: str) -> None:
        ray.get(self.actors[0].save_model.remote(output_dir))

    def evaluate(self) -> None:
        eval_start = time.perf_counter()

        sampling_params = self.sampling_params.clone()
        sampling_params.temperature = 0.0
        sampling_params.n = 1

        results = self.generate(list(self.eval_dataset["prompt"]), sampling_params)

        scores = defaultdict(list)
        for completion_texts, args in zip(
            results["completion_texts"], self.eval_dataset["reward_func_args"]
        ):
            for k, v in self.reward_func(completion_texts[0], **args).items():
                scores[k].append(v)
        mean_scores = {k: np.mean(v).item() for k, v in scores.items()}

        completion_lengths = np.array([
            [len(token_ids) for token_ids in choices_token_ids]
            for choices_token_ids in results["completion_token_ids"]
        ])
        is_terminated = np.array(results["is_terminated"])
        completion_terminated_lengths = completion_lengths[is_terminated]
        if len(completion_terminated_lengths) == 0:
            completion_mean_terminated_length = float("nan")
            completion_min_terminated_length = float("nan")
            completion_max_terminated_length = float("nan")
        else:
            completion_mean_terminated_length = (
                completion_terminated_lengths.mean().item()
            )
            completion_min_terminated_length = (
                completion_terminated_lengths.min().item()
            )
            completion_max_terminated_length = (
                completion_terminated_lengths.max().item()
            )

        runtime = time.perf_counter() - eval_start

        eval_log = {
            "completions/mean_length": completion_lengths.mean().item(),
            "completions/min_length": completion_lengths.min().item(),
            "completions/max_length": completion_lengths.max().item(),
            "completions/clipped_ratio": np.mean(~is_terminated).item(),
            "completions/mean_terminated_length": completion_mean_terminated_length,
            "completions/min_terminated_length": completion_min_terminated_length,
            "completions/max_terminated_length": completion_max_terminated_length,
            **mean_scores,
            "runtime": runtime,
            "samples_per_second": len(self.eval_dataset) / runtime,
        }
        print(eval_log)

        if self.config.report_to == "wandb":
            wandb.log(
                {f"eval/{key}": value for key, value in eval_log.items()},
                step=self.global_step,
            )

    def train(self) -> None:
        train_start = time.perf_counter()
        steps_per_epoch = len(self.train_loader)
        pbar = tqdm(total=self.total_steps)

        self.warmup_generate()

        if self.config.eval_before_train and self.config.eval_strategy != "no":
            self.evaluate()

        for epoch in range(self.config.num_train_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                step_start = time.perf_counter()

                seeds = [
                    derive_seed(self.config.seed, epoch, batch_idx, i)
                    for i in range(self.config.population_size)
                ]
                current_noise_std = self.noise_std_scheduler.get_value()
                current_lr = self.lr_scheduler.get_value()

                generation_start = time.perf_counter()
                results = self.generate_with_noise(
                    batch["prompt"], self.sampling_params, seeds, current_noise_std
                )
                time_generation = time.perf_counter() - generation_start

                calc_rewards_start = time.perf_counter()
                flat_scores = [
                    self.reward_func(choice_completion_text, **args)
                    for model_completion_texts in results["completion_texts"]
                    for sample_completion_texts, args in zip(
                        model_completion_texts, batch["reward_func_args"]
                    )
                    for choice_completion_text in sample_completion_texts
                ]
                time_calc_rewards = time.perf_counter() - calc_rewards_start
                scores = {
                    key: np.array([score[key] for score in flat_scores]).reshape(
                        len(results["completion_texts"]),
                        len(results["completion_texts"][0]),
                        len(results["completion_texts"][0][0]),
                    )
                    for key in flat_scores[0].keys()
                }

                if self.config.macro_advantage:
                    means = scores["reward"].mean(axis=0, keepdims=True)
                    stds = scores["reward"].std(axis=0, keepdims=True)
                    advantages = (scores["reward"] - means) / (
                        stds + ADVANTAGE_STD_EPSILON
                    )
                    advantages = advantages.mean(axis=(1, 2))
                else:
                    model_rewards = scores["reward"].mean(axis=(1, 2))
                    advantages = (model_rewards - model_rewards.mean()) / (
                        model_rewards.std() + ADVANTAGE_STD_EPSILON
                    )

                update_start = time.perf_counter()
                self.update_all_actors(
                    seeds, advantages.tolist(), current_lr, current_noise_std
                )
                time_update = time.perf_counter() - update_start

                self.noise_std_scheduler.step()
                self.lr_scheduler.step()

                completion_lengths = np.array([
                    [
                        [len(token_ids) for token_ids in sample_token_ids]
                        for sample_token_ids in model_token_ids
                    ]
                    for model_token_ids in results["completion_token_ids"]
                ])
                prompt_lengths = np.array([
                    [len(token_ids) for token_ids in model_token_ids]
                    for model_token_ids in results["prompt_token_ids"]
                ])
                self.train_num_tokens += (
                    prompt_lengths.sum() * self.sampling_params.n
                    + completion_lengths.sum()
                )

                self.global_step += 1

                time_step = time.perf_counter() - step_start
                self.train_time_total += time_step

                if self.global_step % self.config.logging_steps == 0:
                    current_epoch = epoch + (batch_idx + 1) / steps_per_epoch

                    is_terminated = np.array(results["is_terminated"])

                    completion_terminated_lengths = completion_lengths[is_terminated]
                    if len(completion_terminated_lengths) == 0:
                        completion_mean_terminated_length = float("nan")
                        completion_min_terminated_length = float("nan")
                        completion_max_terminated_length = float("nan")
                    else:
                        completion_mean_terminated_length = (
                            completion_terminated_lengths.mean().item()
                        )
                        completion_min_terminated_length = (
                            completion_terminated_lengths.min().item()
                        )
                        completion_max_terminated_length = (
                            completion_terminated_lengths.max().item()
                        )

                    mean_scores = {
                        f"{k}/mean": v.mean().item() for k, v in scores.items()
                    }
                    std_scores = {
                        f"{k}/std": v.reshape(v.shape[0], -1).std(axis=0).mean().item()
                        for k, v in scores.items()
                    }

                    reward_stds = (
                        scores["reward"]
                        .reshape(scores["reward"].shape[0], -1)
                        .std(axis=0)
                    )
                    frac_reward_zero_std = np.isclose(reward_stds, 0.0).mean()

                    samples_per_second = (
                        len(batch["prompt"])
                        * self.config.population_size
                        * self.sampling_params.n
                        / time_step
                    )

                    train_log = {
                        "noise_std": current_noise_std,
                        "learning_rate": current_lr,
                        "num_tokens": self.train_num_tokens.item(),
                        "completions/mean_length": completion_lengths.mean().item(),
                        "completions/min_length": completion_lengths.min().item(),
                        "completions/max_length": completion_lengths.max().item(),
                        "completions/clipped_ratio": np.mean(~is_terminated).item(),
                        "completions/mean_terminated_length": completion_mean_terminated_length,
                        "completions/min_terminated_length": completion_min_terminated_length,
                        "completions/max_terminated_length": completion_max_terminated_length,
                        **mean_scores,
                        **std_scores,
                        "frac_reward_zero_std": frac_reward_zero_std.item(),
                        "epoch": current_epoch,
                        "samples_per_second": samples_per_second,
                        "time/generation": time_generation,
                        "time/calc_rewards": time_calc_rewards,
                        "time/update": time_update,
                        "time/step": time_step,
                        "time/total": self.train_time_total,
                    }
                    print(train_log)

                    if self.config.report_to == "wandb":
                        wandb.log(
                            {f"train/{key}": value for key, value in train_log.items()},
                            step=self.global_step,
                        )

                if (
                    self.config.eval_strategy == "steps"
                    and self.config.eval_steps
                    and self.global_step % self.config.eval_steps == 0
                ):
                    self.evaluate()

                if self.config.save_strategy == "steps" and self.config.save_steps:
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(self.global_step)

                pbar.update(1)

            if self.config.eval_strategy == "epoch":
                self.evaluate()

            if self.config.save_strategy == "epoch":
                self.save_checkpoint(self.global_step)

        print({"train_runtime": time.perf_counter() - train_start})

        if self.config.report_to == "wandb":
            wandb.finish()

        pbar.close()
