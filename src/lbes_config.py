from dataclasses import dataclass

from es_config import ESConfig


@dataclass(kw_only=True)
class LBESConfig(ESConfig):
    # SamplingParams
    n: int = 8
    temperature: float = 1.0
    # LBES specific parameters
    flat_corrcoef: bool = False
    top_k_prompt_reward_var: int | None = None
    top_k_rewards: int | None = None
    bottom_k_rewards: int | None = None
    exclude_same_rewards_samples: bool = False
    flat_top_k_rewards: int | None = None
    flat_bottom_k_rewards: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.top_k_prompt_reward_var is not None
            and self.top_k_prompt_reward_var <= 0
        ):
            raise ValueError("top_k_prompt_reward_var must be greater than 0")
        if self.top_k_rewards is not None and self.top_k_rewards <= 0:
            raise ValueError("top_k_rewards must be greater than 0")
        if self.bottom_k_rewards is not None and self.bottom_k_rewards <= 0:
            raise ValueError("bottom_k_rewards must be greater than 0")
        if (self.data_batch_size * self.n) < 2:
            raise ValueError("data_batch_size * n must be at least 2 for PBES")
        if not self.flat_corrcoef and self.n < 2:
            raise ValueError("n must be at least 2 when flat_corrcoef is False")
        if self.top_k_rewards is not None and self.top_k_rewards <= 0:
            raise ValueError("top_k_rewards must be greater than 0")
        if self.bottom_k_rewards is not None and self.bottom_k_rewards <= 0:
            raise ValueError("bottom_k_rewards must be greater than 0")
        if self.flat_top_k_rewards is not None and self.flat_top_k_rewards <= 0:
            raise ValueError("flat_top_k_rewards must be greater than 0")
        if self.flat_bottom_k_rewards is not None and self.flat_bottom_k_rewards <= 0:
            raise ValueError("flat_bottom_k_rewards must be greater than 0")
