"""This module implements the ArbitraryCPhaseGate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_sequence


class ArbitraryCPhaseGate(
    Gate,
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

        dim = self.dim
        rows = []
        for r in range(dim):
            row = ['0'] * dim
            row[r] = 'e^(i*t0)' if r == dim - 1 else '1'
            rows.append('[' + ','.join(row) + ']')

        self._expr = _UnitaryExpression(
            'ArbitraryCPhase_{}<{}>(t0) {{ [{}] }}'.format(
                '_'.join(str(r) for r in self._radixes),
                ','.join(str(r) for r in self._radixes),
                ','.join(rows),
            ),
        )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        U = np.identity(self.dim, dtype=np.complex128)
        U[-1, -1] = np.exp(1j * params[0])
        return UnitaryMatrix(U)

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        p = env_matrix[-1, -1]
        return [-np.arctan2(p.imag, p.real)]

    def __eq__(self, other: object) -> bool:
        """Check if `self` equals `other`."""
        if not isinstance(other, ArbitraryCPhaseGate):
            return False
        return self.radixes == other.radixes

    def __hash__(self) -> int:
        return hash(self.radixes)
