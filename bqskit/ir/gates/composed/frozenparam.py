"""This module implements the FrozenParameterGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class FrozenParameterGate(
        ComposedGate, LocallyOptimizableUnitary, DifferentiableUnitary,
):
    """A composed gate which fixes some parameters of another gate."""

    def __init__(self, gate: Gate, frozen_params: dict[int, float]) -> None:
        """
        Create a gate which fixes some of the parameters it takes.

        Args:
            gate (Gate): The Gate to fix the parameters of.
            frozen_params (dict[int, float]): A dictionary mapping parameters
                indices to the fixed value they should be.

        Raises:
            ValueError: If any of the `frozen_params` indices are greater
                than the number of parameters `gate` takes or less than 0
                or if the total amount of `frozen_params` is larger than
                the number of parameters `gate` takes.
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate, got %s.' % type(gate))
        if not isinstance(frozen_params, dict):
            raise TypeError(
                'Expected dict for frozen_params, '
                'got %s.' % type(frozen_params),
            )
        if not len(frozen_params) <= gate.num_params:
            raise ValueError(
                'Too many fixed parameters specified, expected at most'
                ' %d, got %d' % (gate.num_params, len(frozen_params)),
            )
        keys = list(frozen_params.keys())
        values = list(frozen_params.values())
        if not all(isinstance(p, int) for p in keys):
            fail_idx = [isinstance(p, int) for p in keys].index(False)
            raise TypeError(
                'Expected frozen_params keys to be int, got %s.'
                % type(keys[fail_idx]),
            )
        if not all(isinstance(p, (int, float)) for p in values):
            typechecks = [isinstance(p, (int, float)) for p in values]
            fail_idx = typechecks.index(False)
            raise TypeError(
                'Expected frozen_params values to be float, got %s.'
                % type(values[fail_idx]),
            )
        if not all(0 <= p < gate.num_params for p in keys):
            fail_idx = [
                0 <= p < gate.num_params
                for p in keys
            ].index(False)
            raise ValueError(
                'Expected parameter index to be non-negative integer'
                ' < %d, got %d.' % (gate.num_params, keys[fail_idx]),
            )

        self.gate = gate
        self._num_params = gate.num_params - len(frozen_params)
        self._num_qudits = gate.num_qudits
        self._radixes = gate.radixes
        self._qasm_name = self.gate.qasm_name
        self.frozen_params = frozen_params
        self.unfixed_param_idxs = [
            i for i in range(gate.num_params)
            if i not in self.frozen_params.keys()
        ]

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        return self.gate.get_unitary(self.get_full_params(params))

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        grads = self.gate.get_grad(  # type: ignore
            self.get_full_params(params),
        )
        return grads[self.unfixed_param_idxs, :, :]

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        params = self.gate.optimize(env_matrix)  # type: ignore
        return [
            p for i, p in enumerate(params)
            if i in self.unfixed_param_idxs
        ]

    @property
    def name(self) -> str:
        """The name of the gate, see Gate for more info."""
        return '%s(%s, %s)' % (
            self.__class__.__name__,
            self.gate.name,
            str(self.frozen_params),
        )

    def get_qasm_gate_def(self) -> str:
        """Returns the qasm gate def for this gate, see Gate for more info."""
        return self.gate.get_qasm_gate_def()

    def get_full_params(self, params: Sequence[float]) -> list[float]:
        """Returns the full parameter list for the underlying gate."""
        self.check_parameters(params)
        args = list(params)
        for idx in sorted(self.frozen_params):
            args.insert(idx, self.frozen_params[idx])
        return args


def with_frozen_params(
        self: Gate,
        frozen_params: dict[int, float],
) -> FrozenParameterGate:
    return FrozenParameterGate(self, frozen_params)


def with_all_frozen_params(
    self: Gate,
    params: list[float],
) -> FrozenParameterGate:
    return FrozenParameterGate(self, {i: x for i, x in enumerate(params)})


Gate.with_frozen_params = with_frozen_params
Gate.with_all_frozen_params = with_all_frozen_params
