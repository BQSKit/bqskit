"""This module implements a general Diagonal Gate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class DiagonalGate(
    Gate,
    CachedClass,
    LocallyOptimizableUnitary,
):
    """
    A gate representing a general diagonal unitary.

    The top-left element is fixed to 1, and the rest are set to exp(i * theta).

    This gate is used to optimize the Block ZXZ decomposition of a unitary.
    """

    _qasm_name = 'diag'

    def __init__(
        self,
        num_qudits: int = 2,
    ):
        self._num_qudits = num_qudits
        # 1 parameter per diagonal element, removing one for global phase
        self._num_params = 2 ** num_qudits - 1

        # TODO/OQ: fix
        # A raw QGL matrix literal only ever infers a single flat qudit
        # from its dimension, so the multi-qudit radices are built by
        # embedding each parameterized diagonal entry into a tagged
        # identity instead of parsing a `2**n`x`2**n` literal directly.
        self._expr = _UnitaryExpression.identity(
            'Diagonal', [2] * num_qudits,
        )
        for i in range(1, 2 ** num_qudits):
            sub = _UnitaryExpression(
                'Phase%d(t%d) { [[e^(i*t%d)]] }' % (i, i - 1, i - 1),
            )
            self._expr.embed(sub, i, i)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        mat = np.eye(2 ** self.num_qudits, dtype=np.complex128)

        for i in range(1, 2 ** self.num_qudits):
            mat[i][i] = np.exp(1j * params[i - 1])

        return UnitaryMatrix(mat)

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        thetas = [0.0] * self.num_params

        base = env_matrix[0, 0]
        if base == 0:
            base = np.max(env_matrix[0, :])

        for i in range(1, 2 ** self.num_qudits):
            # Optimize each angle independently
            a = np.angle(env_matrix[i, i] / base)
            thetas[i - 1] = -1 * a

        return thetas
