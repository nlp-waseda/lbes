import math


class Scheduler:
    def __init__(self, initial_value: float, total_steps: int):
        self.initial_value = initial_value
        self.total_steps = total_steps
        self.current_step = 0

    def get_value(self) -> float:
        raise NotImplementedError

    def step(self) -> None:
        self.current_step += 1


class ConstantScheduler(Scheduler):
    def get_value(self) -> float:
        return self.initial_value


class LinearScheduler(Scheduler):
    def __init__(
        self, initial_value: float, total_steps: int, final_value: float = 0.0
    ):
        super().__init__(initial_value, total_steps)
        self.final_value = final_value

    def get_value(self) -> float:
        if self.current_step >= self.total_steps:
            return self.final_value

        progress = self.current_step / self.total_steps
        return self.initial_value + (self.final_value - self.initial_value) * progress


class CosineScheduler(Scheduler):
    def __init__(
        self, initial_value: float, total_steps: int, final_value: float = 0.0
    ):
        super().__init__(initial_value, total_steps)
        self.final_value = final_value

    def get_value(self) -> float:
        if self.current_step >= self.total_steps:
            return self.final_value

        progress = self.current_step / self.total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_value + (self.initial_value - self.final_value) * cosine_decay


class ExponentialScheduler(Scheduler):
    def __init__(
        self, initial_value: float, total_steps: int, decay_rate: float = 0.95
    ):
        super().__init__(initial_value, total_steps)
        self.decay_rate = decay_rate

    def get_value(self) -> float:
        return self.initial_value * (self.decay_rate**self.current_step)


class PolynomialScheduler(Scheduler):
    def __init__(
        self,
        initial_value: float,
        total_steps: int,
        final_value: float = 0.0,
        power: float = 1.0,
    ):
        super().__init__(initial_value, total_steps)
        self.final_value = final_value
        self.power = power

    def get_value(self) -> float:
        if self.current_step >= self.total_steps:
            return self.final_value

        progress = self.current_step / self.total_steps
        decay_factor = (1 - progress) ** self.power
        return self.final_value + (self.initial_value - self.final_value) * decay_factor


def create_scheduler(
    scheduler_type: str, initial_value: float, total_steps: int, **kwargs
) -> Scheduler:
    if scheduler_type == "constant":
        return ConstantScheduler(initial_value, total_steps)
    elif scheduler_type == "linear":
        final_value = kwargs.get("final_value", 0.0)
        return LinearScheduler(initial_value, total_steps, final_value)
    elif scheduler_type == "cosine":
        final_value = kwargs.get("final_value", 0.0)
        return CosineScheduler(initial_value, total_steps, final_value)
    elif scheduler_type == "exponential":
        decay_rate = kwargs.get("decay_rate", 0.95)
        return ExponentialScheduler(initial_value, total_steps, decay_rate)
    elif scheduler_type == "polynomial":
        final_value = kwargs.get("final_value", 0.0)
        power = kwargs.get("power", 1.0)
        return PolynomialScheduler(initial_value, total_steps, final_value, power)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
