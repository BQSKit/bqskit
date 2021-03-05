"""This module implements the StateVector class."""
from __future__ import annotations

from typing import Any
from typing import Sequence
from typing import Union

import numpy as np


class StateVector:
    """The StateVector class."""

    def __init__(self, vec: VectorLike, radixes: Sequence[int] = []) -> None:
        """
        Constructs a StateVector with the supplied vector.

        Args:
            vec (VectorLike): The vector.

            radixes (Sequence[int]): A sequence with its length equal to
                the number of qudits this StateVector represents. Each
                element specifies the base, number of orthogonal states,
                for the corresponding qudit. By default, the constructor
                will attempt to calculate `radixes` from `vec`.

        Raises:
            TypeError: If `radixes` is not specified and the constructor
                cannot determine `radixes`.
        """


VectorLike = Union[StateVector, np.ndarray, Sequence[Any]]
