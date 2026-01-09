import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.v1.worker.gpu_worker import Worker

from seed_utils import derive_seed, set_seed


class WorkerExtension:
    def add_noise(self: Worker, seed: int, noise_std: float) -> None:
        set_seed(derive_seed(seed, get_tensor_model_parallel_rank()))
        with torch.no_grad():
            for _, param in self.model_runner.model.named_parameters():
                param.add_(torch.randn_like(param), alpha=noise_std)

    def remove_noise(self: Worker, seed: int, noise_std: float) -> None:
        set_seed(derive_seed(seed, get_tensor_model_parallel_rank()))
        with torch.no_grad():
            for _, param in self.model_runner.model.named_parameters():
                param.sub_(torch.randn_like(param), alpha=noise_std)

    def update(
        self: Worker,
        seeds: list[int],
        advantages: list[float],
        lr: float,
        noise_std: float,
    ) -> None:
        base_step_size = lr / (len(seeds) * noise_std)

        for seed, advantage in zip(seeds, advantages):
            if advantage == 0.0 or advantage != advantage:  # Check for NaN
                continue
            step_size = base_step_size * advantage

            set_seed(derive_seed(seed, get_tensor_model_parallel_rank()))
            with torch.no_grad():
                for _, param in self.model_runner.model.named_parameters():
                    param.add_(torch.randn_like(param), alpha=step_size)

    def save_safetensors(self: Worker, output_dir: str, max_shard_size="5GB") -> None:
        def parse_size(v):
            if isinstance(v, int):
                return v
            s = str(v).strip().upper()
            units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
            for u in ["GB", "MB", "KB", "B"]:
                if s.endswith(u):
                    return int(float(s[: -len(u)].strip()) * units[u])
            return int(s)

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        limit = parse_size(max_shard_size)

        sd = self.model_runner.model.state_dict()
        sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}

        shards = []
        current, cur_bytes = {}, 0
        total_bytes = 0
        for k, v in sd.items():
            b = v.numel() * v.element_size()
            total_bytes += b
            if current and cur_bytes + b > limit:
                shards.append(current)
                current, cur_bytes = {}, 0
            current[k] = v
            cur_bytes += b
        if current:
            shards.append(current)

        n = max(1, len(shards))
        weight_map = {}
        for i, shard in enumerate(shards, 1):
            if n == 1:
                fname = "model.safetensors"
            else:
                fname = f"model-{i:05d}-of-{n:05d}.safetensors"

            save_file(shard, output_path / fname)
            for k in shard.keys():
                weight_map[k] = fname

        if n > 1:
            index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
            index_path = output_path / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index, indent=2))
