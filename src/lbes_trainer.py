import time

import numpy as np
import ray
import wandb
from numpy.typing import NDArray
from tqdm import tqdm

from es_trainer import ADVANTAGE_STD_EPSILON, ESTrainer
from lbes_config import LBESConfig
from seed_utils import derive_seed


class LBESTrainer(ESTrainer):
    """Likelihood-Based Evolution Strategies Trainer"""

    def __init__(self, config: LBESConfig) -> None:
        self.config = config
        self.setup()

    def warmup_generate(self) -> None:
        prompts = [[{"role": "user", "content": "Warmup"}]]
        sampling_params = self.sampling_params.clone()
        sampling_params.n = 1
        sampling_params.max_tokens = 1
        sampling_params.prompt_logprobs = 1
        sampling_params.flat_logprobs = True
        ray.get([
            actor.generate.remote(prompts, sampling_params) for actor in self.actors
        ])

    def calc_mean_logprobs_with_noise(
        self,
        prompt_token_ids: list[list[int]],  # (batch_size, )
        completion_token_ids: list[list[list[int]]],  # (batch_size, n)
        seeds: list[int],
        current_noise_std: float,
    ) -> list[list[list[float]]]:  # (population_size, batch_size, n)
        num_actors = len(self.actors)
        num_seeds = len(seeds)
        mean_logprobs = []
        pos = 0
        while pos < num_seeds:
            num_noises = min(num_actors, num_seeds - pos)
            noise_actors = self.actors[:num_noises]

            ray.get([
                actor.add_noise.remote(seed, current_noise_std)
                for actor, seed in zip(noise_actors, seeds[pos : pos + num_noises])
            ])

            results = ray.get([
                actor.calc_mean_logprobs.remote(prompt_token_ids, completion_token_ids)
                for actor in noise_actors
            ])
            mean_logprobs.extend(results)

            ray.get([
                actor.remove_noise.remote(seed, current_noise_std)
                for actor, seed in zip(noise_actors, seeds[pos : pos + num_noises])
            ])

            pos += num_noises

        # (population_size, batch_size, n)
        return mean_logprobs

    def filter_batch_by_reward(
        self,
        rewards: NDArray[np.float64],  # (batch_size, n)
        prompt_token_ids: list[list[int]],  # (batch_size, )
        completion_token_ids: list[list[list[int]]],  # (batch_size, n)
    ) -> tuple[
        list[list[list[int]]],  # (filtered_batch_size, n)
        list[list[int]],  # (filtered_batch_size, )
        list[list[list[int]]],  # (filtered_batch_size, n)
        dict[str, float],
    ]:
        if all([
            self.config.top_k_prompt_reward_var is None,
            self.config.top_k_rewards is None,
            self.config.bottom_k_rewards is None,
            not self.config.exclude_same_rewards_samples,
            self.config.flat_top_k_rewards is None,
            self.config.flat_bottom_k_rewards is None,
        ]):
            return (
                rewards.tolist(),
                prompt_token_ids,
                completion_token_ids,
                {"retained_prompt_ratio": 1.0, "retained_reward_ratio": 1.0},
            )

        original_batch_size = rewards.shape[0]
        original_num_choices = rewards.shape[1]

        if self.config.top_k_prompt_reward_var is not None:
            k = min(self.config.top_k_prompt_reward_var, original_batch_size)
            prompt_reward_vars = rewards.var(axis=1)

            rng = np.random.default_rng(
                derive_seed(
                    self.config.seed,
                    self.global_step,
                    "prompt_reward_var_tiebreak",
                )
            )
            tie_breaker = rng.random(size=prompt_reward_vars.shape[0])
            order = np.lexsort((tie_breaker, prompt_reward_vars))
            keep_indices = np.sort(order[-k:])

            rewards = rewards[keep_indices]
            prompt_token_ids = [prompt_token_ids[i] for i in keep_indices.tolist()]
            completion_token_ids = [
                completion_token_ids[i] for i in keep_indices.tolist()
            ]

        filtered_rewards = []
        filtered_prompt_token_ids = []
        filtered_completion_token_ids = []
        for sample_idx in range(rewards.shape[0]):
            sample_rewards = rewards[sample_idx]

            selected_indices = set()

            if self.config.top_k_rewards is not None:
                top_k = min(self.config.top_k_rewards, len(sample_rewards))
                top_k_indices = np.argsort(sample_rewards)[-top_k:]
                selected_indices.update(top_k_indices.tolist())

            if self.config.bottom_k_rewards is not None:
                bottom_k = min(self.config.bottom_k_rewards, len(sample_rewards))
                bottom_k_indices = np.argsort(sample_rewards)[:bottom_k]
                selected_indices.update(bottom_k_indices.tolist())

            if (
                self.config.top_k_rewards is None
                and self.config.bottom_k_rewards is None
            ):
                selected_indices.update(range(len(sample_rewards)))

            selected_indices = sorted(selected_indices)

            if (
                self.config.exclude_same_rewards_samples
                and len(selected_indices) > 1
                and np.allclose(
                    sample_rewards[selected_indices[1:]],
                    [sample_rewards[selected_indices[0]]] * (len(selected_indices) - 1),
                )
            ):
                continue

            filtered_rewards.append(sample_rewards[selected_indices])
            filtered_prompt_token_ids.append(prompt_token_ids[sample_idx])
            filtered_completion_token_ids.append([
                completion_token_ids[sample_idx][i] for i in selected_indices
            ])

        if (
            self.config.flat_top_k_rewards is not None
            or self.config.flat_bottom_k_rewards is not None
        ):
            if not filtered_rewards:
                return (
                    [],
                    [],
                    [],
                    {
                        "retained_prompt_ratio": 0.0,
                        "retained_reward_ratio": 0.0,
                    },
                )
            sample_sizes = [len(sample_rewards) for sample_rewards in filtered_rewards]
            total_choices = sum(sample_sizes)

            cumulative_sizes = np.cumsum(sample_sizes)
            all_filtered_rewards = np.concatenate(filtered_rewards)
            selected_masks = [
                np.zeros(sample_size, dtype=bool) for sample_size in sample_sizes
            ]

            def mark_indices(flat_indices: np.ndarray) -> None:
                for idx in flat_indices:
                    sample_idx = np.searchsorted(cumulative_sizes, idx, side="right")
                    prev_total = (
                        cumulative_sizes[sample_idx - 1] if sample_idx > 0 else 0
                    )
                    local_idx = idx - prev_total
                    selected_masks[sample_idx][local_idx] = True

            rng = np.random.default_rng(
                derive_seed(self.config.seed, self.global_step, "flat_reward_tiebreak")
            )
            tie_breaker = rng.random(size=all_filtered_rewards.shape[0])
            reward_order = np.lexsort((tie_breaker, all_filtered_rewards))

            if self.config.flat_top_k_rewards is not None:
                flat_top_k = min(self.config.flat_top_k_rewards, total_choices)
                top_k_indices = reward_order[-flat_top_k:]
                mark_indices(top_k_indices)

            if self.config.flat_bottom_k_rewards is not None:
                flat_bottom_k = min(self.config.flat_bottom_k_rewards, total_choices)
                bottom_k_indices = reward_order[:flat_bottom_k]
                mark_indices(bottom_k_indices)

            new_filtered_rewards = []
            new_filtered_prompt_token_ids = []
            new_filtered_completion_token_ids = []

            for sample_idx, sample_rewards in enumerate(filtered_rewards):
                mask = selected_masks[sample_idx]
                if not mask.any():
                    continue

                selected_indices = np.flatnonzero(mask)

                new_filtered_rewards.append(sample_rewards[selected_indices])
                new_filtered_prompt_token_ids.append(
                    filtered_prompt_token_ids[sample_idx]
                )
                new_filtered_completion_token_ids.append([
                    filtered_completion_token_ids[sample_idx][i]
                    for i in selected_indices
                ])

            filtered_rewards = new_filtered_rewards
            filtered_prompt_token_ids = new_filtered_prompt_token_ids
            filtered_completion_token_ids = new_filtered_completion_token_ids

        return (
            [r.tolist() for r in filtered_rewards],
            filtered_prompt_token_ids,
            filtered_completion_token_ids,
            {
                "retained_prompt_ratio": len(filtered_rewards) / original_batch_size,
                "retained_reward_ratio": sum(len(r) for r in filtered_rewards)
                / (original_batch_size * original_num_choices),
            },
        )

    def calc_corrcoef(
        self,
        rewards: list[list[float]],  # (batch_size, n)
        logprobs: list[list[list[float]]],  # (population_size, batch_size, n)
        flatten: bool = False,
    ) -> NDArray[np.float64]:
        if flatten:
            # (batch_size * n,)
            rewards_flat = np.array([
                r for sample_rewards in rewards for r in sample_rewards
            ])
            # (population_size, batch_size * n)
            logprobs_flat = np.array([
                [lp for sample_logprobs in model_logprobs for lp in sample_logprobs]
                for model_logprobs in logprobs
            ])

            # scalar
            mean_reward = np.mean(rewards_flat)
            # (population_size, 1)
            mean_logprobs = np.mean(logprobs_flat, axis=1, keepdims=True)

            # (batch_size * n,)
            centered_rewards = rewards_flat - mean_reward
            # (population_size, batch_size * n)
            centered_logprobs = logprobs_flat - mean_logprobs

            # scalar
            normed_rewards = np.sqrt(np.sum(centered_rewards**2))
            # (population_size, 1)
            normed_logprobs = np.sqrt(
                np.sum(centered_logprobs**2, axis=1, keepdims=True)
            )

            # (population_size,)
            numerator = np.sum(centered_rewards * centered_logprobs, axis=1)
            # (population_size,)
            denominator = (normed_rewards * normed_logprobs).squeeze(1)

            # (population_size,)
            correlations = np.where(denominator != 0.0, numerator / denominator, np.nan)

            return correlations
        else:
            # (batch_size, )
            mean_rewards = np.array([
                np.mean(sample_rewards) for sample_rewards in rewards
            ])
            # (population_size, batch_size)
            mean_logprobs = np.array([
                [np.mean(sample_logprobs) for sample_logprobs in model_logprobs]
                for model_logprobs in logprobs
            ])

            # (batch_size, n)
            centered_rewards = [
                np.array(sample_rewards) - mean_reward
                for sample_rewards, mean_reward in zip(rewards, mean_rewards)
            ]
            # (population_size, batch_size, n)
            centered_logprobs = [
                [
                    np.array(sample_logprobs) - mean_logprob
                    for sample_logprobs, mean_logprob in zip(
                        model_logprobs, mean_logprobs_model
                    )
                ]
                for model_logprobs, mean_logprobs_model in zip(logprobs, mean_logprobs)
            ]

            # (batch_size, )
            rewards_norm = np.sqrt([np.sum(cr**2) for cr in centered_rewards])
            # (population_size, batch_size, )
            logprobs_norm = np.array([
                [np.sqrt(np.sum(cls**2)) for cls in model_centered_logprobs]
                for model_centered_logprobs in centered_logprobs
            ])

            # (population_size, batch_size)
            numerator = np.array([
                [
                    np.sum(
                        centered_rewards[sample_idx]
                        * centered_logprobs_model[sample_idx]
                    )
                    for sample_idx in range(len(centered_rewards))
                ]
                for centered_logprobs_model in centered_logprobs
            ])
            # (population_size, batch_size)
            denominator = rewards_norm * logprobs_norm

            # (population_size, batch_size)
            correlations = np.where(denominator != 0.0, numerator / denominator, np.nan)

            mean_correlations = np.nanmean(correlations, axis=1)  # (population_size,)

            return mean_correlations

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

                generation_start = time.perf_counter()
                results = self.generate(batch["prompt"], self.sampling_params)
                time_generation = time.perf_counter() - generation_start

                calc_rewards_start = time.perf_counter()
                flat_scores = [
                    self.reward_func(choice_completion_text, **args)
                    for sample_completion_texts, args in zip(
                        results["completion_texts"], batch["reward_func_args"]
                    )
                    for choice_completion_text in sample_completion_texts
                ]
                time_calc_rewards = time.perf_counter() - calc_rewards_start
                scores = {
                    key: np.array([score[key] for score in flat_scores]).reshape(
                        len(results["completion_texts"]),
                        len(results["completion_texts"][0]),
                    )
                    for key in flat_scores[0].keys()
                }

                (
                    filtered_rewards,
                    filtered_prompt_token_ids,
                    filtered_completion_token_ids,
                    filter_stats,
                ) = self.filter_batch_by_reward(
                    scores["reward"],
                    results["prompt_token_ids"],
                    results["completion_token_ids"],
                )

                if not filtered_prompt_token_ids:
                    print("Skipping step because all samples were filtered out")
                    pbar.update(1)
                    continue

                seeds = [
                    derive_seed(self.config.seed, epoch, batch_idx, i)
                    for i in range(self.config.population_size)
                ]
                current_noise_std = self.noise_std_scheduler.get_value()
                current_lr = self.lr_scheduler.get_value()

                calc_logprobs_start = time.perf_counter()
                # (population_size, batch_size, n)
                mean_logprobs = self.calc_mean_logprobs_with_noise(
                    filtered_prompt_token_ids,
                    filtered_completion_token_ids,
                    seeds,
                    current_noise_std,
                )
                time_calc_logprobs = time.perf_counter() - calc_logprobs_start

                corrcoef = self.calc_corrcoef(
                    filtered_rewards,
                    mean_logprobs,
                    flatten=self.config.flat_corrcoef,
                )

                advantages = (corrcoef - np.nanmean(corrcoef)) / (
                    np.nanstd(corrcoef) + ADVANTAGE_STD_EPSILON
                )

                update_start = time.perf_counter()
                self.update_all_actors(
                    seeds, advantages.tolist(), current_lr, current_noise_std
                )
                time_update = time.perf_counter() - update_start

                self.noise_std_scheduler.step()
                self.lr_scheduler.step()

                completion_lengths = np.array([
                    [len(token_ids) for token_ids in choices_token_ids]
                    for choices_token_ids in results["completion_token_ids"]
                ])
                prompt_token_lengths = np.array([
                    len(token_ids) for token_ids in results["prompt_token_ids"]
                ])
                self.train_num_tokens += (
                    prompt_token_lengths.sum() * self.sampling_params.n
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
                        f"{k}/std": v.std(axis=1).mean().item()
                        for k, v in scores.items()
                    }

                    reward_stds = scores["reward"].std(axis=1)
                    frac_reward_zero_std = np.isclose(reward_stds, 0.0).mean()

                    samples_per_second = (
                        len(batch["prompt"]) * self.sampling_params.n / time_step
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
                        **filter_stats,
                        "frac_reward_zero_std": frac_reward_zero_std.item(),
                        "epoch": current_epoch,
                        "samples_per_second": samples_per_second,
                        "time/generation": time_generation,
                        "time/calc_rewards": time_calc_rewards,
                        "time/calc_logprobs": time_calc_logprobs,
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
