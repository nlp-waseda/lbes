import hashlib
import random
from typing import Any

import numpy as np
import torch

MAX_SEED_VALUE = 2**31


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derive_seed(base_seed: int, *args: Any) -> int:
    combined = f"{base_seed}_{'_'.join(str(arg) for arg in args)}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()

    seed = int.from_bytes(hash_bytes[:4], byteorder="big") % MAX_SEED_VALUE
    return seed
