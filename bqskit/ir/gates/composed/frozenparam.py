"""This module implements the FrozenParameterGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number


class FrozenParameterGate(
    ComposedGate,
    LocallyOptimizableUnitary,
    DifferentiableUnitary,
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
        if not all(is_integer(p) for p in keys):
            fail_idx = [is_integer(p) for p in keys].index(False)
            raise TypeError(
                'Expected frozen_params keys to be int, got %s.'
                % type(keys[fail_idx]),
            )
        if not all(is_real_number(p) for p in values):
            typechecks = [is_real_number(p) for p in values]
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
        self.frozen_params = frozen_params
        self.unfixed_param_idxs = [
            i for i in range(gate.num_params)
            if i not in self.frozen_params.keys()
        ]
        self._name = '{}({}, {})'.format(
            self.__class__.__name__,
            self.gate.name,
            str(self.frozen_params),
        )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self.gate.get_unitary(self.get_full_params(params))

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        grads = self.gate.get_grad(self.get_full_params(params))  # type: ignore
        return grads[self.unfixed_param_idxs, :, :]

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        f_params = self.get_full_params(params)

        utry, grads = self.gate.get_unitary_and_grad(f_params)  # type: ignore
        return utry, grads[self.unfixed_param_idxs, :, :]

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        params = self.gate.optimize(env_matrix)  # type: ignore
        return [
            p for i, p in enumerate(params)
            if i in self.unfixed_param_idxs
        ]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, FrozenParameterGate)
            and self.gate == other.gate
            and self.frozen_params == other.frozen_params
        )

    def __hash__(self) -> int:
        return hash((self.gate, tuple(self.frozen_params.items())))

    @property
    def qasm_name(self) -> str:
        """The qasm command for this gate, see Gate for more info."""
        return self.gate.qasm_name

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm def for this gate, see :class:Gate for more."""
        return self.gate.get_qasm_gate_def()

    def get_full_params(self, params: RealVector) -> list[float]:
        """
        Returns the full parameter list for the underlying gate.

        Args:
            params (RealVector): The parameters to the gate.

        Returns:
            list[float]: The full parameters to the underlying gate.
        """
        self.check_parameters(params)
        args = list(params)
        for idx in sorted(self.frozen_params):
            args.insert(idx, self.frozen_params[idx])
        return args


def with_frozen_params(
        self: Gate,
        frozen_params: dict[int, float],
) -> FrozenParameterGate:
    """
    Freeze some of a gate's parameters so they don't change from optimization.

    Args:
        frozen_params (dict[int, float]): A map from parameter indices to
            parameters values. If i in frozen_params, then this will freeze
            the i-th parameter to the value given by frozen_params[i].

    Returns:
        FrozenParameterGate: The gate with some parameters frozen.
    """
    return FrozenParameterGate(self, frozen_params)


def with_all_frozen_params(
    self: Gate,
    params: list[float],
) -> FrozenParameterGate:
    """
    Freeze all of a gate's parameters so they don't change from optimization.

    Args:
        params (list[float]): The values to set and freeze all parameters to.

    Returns:
        FrozenParameterGate: The gate with all parameters frozen.
    """
    return FrozenParameterGate(self, {i: x for i, x in enumerate(params)})


Gate.with_frozen_params = with_frozen_params
Gate.with_all_frozen_params = with_all_frozen_params
