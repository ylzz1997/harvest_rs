from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class HarvestOption:
    f0_floor: float
    f0_ceil: float
    frame_period: float

    def __init__(self, f0_floor: float = 90.0, f0_ceil: float = 1600.0, frame_period: float = 10.0) -> None: ...


def harvest(
    x: NDArray[np.float32] | NDArray[np.float64],  # shape: [SEQ]
    fs: int,
    option: Optional[HarvestOption] = ...,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...