from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt


class RunnerResults:
    """The result of running a Circuit through a CircuitRunner."""

    def __init__(
        self,
        num_qudits: int,
        radixes: Sequence[int],
        probs: Sequence[float],
    ) -> None:
        """Construct a RunnerResults object."""
        self.num_qudits = num_qudits
        self.radixes = radixes
        self.probs = np.array(probs)

    def get_counts(self, shots: int) -> npt.NDArray[np.int64]:
        return np.asarray(np.multiply(shots, self.probs), np.int64)

    def __str__(self) -> str:
        pass
