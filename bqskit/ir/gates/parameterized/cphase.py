"""This module implements the ArbitraryCPhaseGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_sequence


class ArbitraryCPhaseGate(
    Gate,
    DifferentiableUnitary,
    CachedClass,
    LocallyOptimizableUnitary,
):
    """A gate representing an arbitrary qudit controlled phase rotation."""

    _num_params = 1

    def __init__(self, radixes: Sequence[int] = []) -> None:
        if len(radixes) == 0:
            radixes = [2, 2]

        if not is_sequence(radixes):
            raise TypeError(
                f'Expected sequence for radixes, got {type(radixes)}.',
            )

        if any(r <= 1 for r in radixes):
            raise TypeError('Invalid radixes, all radixes must be >= 2.')

        self._num_qudits = len(radixes)
        self._radixes = tuple(radixes)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        U = np.identity(self.dim, dtype=np.complex128)
        U[-1, -1] = np.exp(1j * params[0])
        return UnitaryMatrix(U)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)
        dU = np.zeros((1, self.dim, self.dim), dtype=np.complex128)
        dU[-1, -1, -1] = 1j * np.exp(1j * params[0])
        return dU

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        p = env_matrix[-1, -1]
        return [-np.arctan2(p.imag, p.real)]
