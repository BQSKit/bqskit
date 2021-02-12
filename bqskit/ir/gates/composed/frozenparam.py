"""This module implements the FrozenParameterGate."""
from __future__ import annotations
from typing import Sequence

import numpy as np

from bqskit.ir import Gate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class FrozenParameterGate(Gate):
    """A composed gate which fixes some parameters of another gate."""

    def __init__(self, gate: Gate, fixed_params: dict[int, float]) -> None:
        """Create a gate which fixes some of the parameters it takes.
        
        Args:
            gate (Gate): The Gate to fix the parameters of.
            fixed_params (dict[int, float]): A dictionary mapping parameters
                indices to the fixed value they should be.
        
        Raises:
            ValueError: if any of the `fixed_params` indices are greater than the
                number of parameters `gate` takes.        
        """
        self.num_params = gate.num_params - len(fixed_params)
        self.size = gate.size
        self.radixes = gate.radixes
        self.fixed_params = fixed_params
        self.gate = gate

    def get_unitary(self, params: Sequence[float]) -> UnitaryMatrix:
        if params is None or len(params) != self.num_params:
            raise ValueError(f"{self.name} takes {self.num_params} parameters.")

        args = list(params)
        for idx, val in sorted(self.fixed_params.items(), key=lambda k, v: k):
            args.insert(idx, val)
        return self.gate.get_unitary(args)
    
    @property
    def name(self):
        return f'{self.__class__.__name__}({self.gate.name})'


def with_frozen_params(self, frozen_params: dict[int, float]) -> FrozenParameterGate:
    return FrozenParameterGate(self, frozen_params)

def with_all_frozen_params(self, params: list[float]) -> FrozenParameterGate:
    if self.get_num_params() != len(params):
        raise ValueError("Invalid parameter list.")
    return FrozenParameterGate(self, {i: x for i, x in enumerate(params)})

Gate.with_frozen_params = with_frozen_params
Gate.with_all_frozen_params = with_all_frozen_params

