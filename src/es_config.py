from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True)
class ESConfig:
    # EngineArgs
    model: str
    load_format: str = "auto"
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    # SamplingParams
    n: int = 1
    temperature: float = 0.0
    max_tokens: int = 2048
    # Other
    output_dir: str
    train_file: str
    eval_file: str | None = None
    reward_func_path: str
    data_batch_size: int = 64
    population_size: int = 32
    noise_std: float = 1.0e-3
    noise_std_scheduler_type: str = "constant"
    noise_std_final: float = 0.0
    learning_rate: float = 5.0e-7
    lr_scheduler_type: str = "constant"
    lr_final: float = 0.0
    num_train_epochs: int = 3
    logging_steps: int = 1
    eval_strategy: Literal["no", "steps", "epoch"] = "no"
    eval_steps: int | None = None
    eval_before_train: bool = True
    save_strategy: Literal["no", "steps", "epoch"] = "no"
    save_steps: int | None = None
    save_total_limit: int = 3
    seed: int = 42
    report_to: str | None = None
    run_name: str | None = None
    macro_advantage: bool = False

    def __post_init__(self) -> None:
        for field_name, field_value in [
            ("data_batch_size", self.data_batch_size),
            ("population_size", self.population_size),
            ("noise_std", self.noise_std),
            ("learning_rate", self.learning_rate),
            ("num_train_epochs", self.num_train_epochs),
            ("logging_steps", self.logging_steps),
            ("save_total_limit", self.save_total_limit),
        ]:
            if field_value <= 0:
                raise ValueError(f"{field_name} must be greater than 0")
        for field_name, field_value in [
            ("noise_std_final", self.noise_std_final),
            ("lr_final", self.lr_final),
        ]:
            if field_value < 0:
                raise ValueError(f"{field_name} must be greater than or equal to 0")
        if self.eval_strategy != "no" and self.eval_file is None:
            raise ValueError("eval_file must be provided if eval_strategy is not 'no'")
        if self.eval_strategy == "steps" and self.eval_steps is None:
            raise ValueError("eval_steps must be provided if eval_strategy is 'steps'")
        if self.save_strategy == "steps" and self.save_steps is None:
            raise ValueError("save_steps must be provided if save_strategy is 'steps'")
