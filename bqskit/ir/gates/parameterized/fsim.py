"""This module implements the FSIMGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class FSIMGate(
    Gate,
    CachedClass,
):
    """
    Google's FSIM Gate.

    Contains all two qubit interactions that preserve excitations,
    up to single-qubit rotations and global phase.

    It is given by the following parameterized unitary:

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & \\cos{\\theta} & -i\\sin{\\theta} & 0 \\\\
        0 & -i\\sin{\\theta} & \\cos{\\theta} & 0 \\\\
        0 & 0 & 0 & e^{-i\\phi} \\\\
        \\end{pmatrix}

    References:
        https://quantumai.google/reference/python/cirq/ops/FSimGate
    """

    _num_qudits = 2
    _num_params = 2
    _qasm_name = 'fsim'
    _expr = _UnitaryExpression(
        'FSIM(t0,t1) { [[1,0,0,0],'
        '[0,cos(t0),~i*sin(t0),0],'
        '[0,~i*sin(t0),cos(t0),0],'
        '[0,0,0,e^(~i*t1)]] }',
    )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0])
        sin = -1j * np.sin(params[0])
        phi = np.exp(-1j * params[1])

        return UnitaryMatrix(
            [
                [1, 0, 0, 0],
                [0, cos, sin, 0],
                [0, sin, cos, 0],
                [0, 0, 0, phi],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`~bqskit.ir.gate.Gate` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0])
        dsin = -1j * np.cos(params[0])
        dphi = -1j * np.exp(-1j * params[1])

        return np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, dcos, dsin, 0],
                    [0, dsin, dcos, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, dphi],
                ],
            ], dtype=np.complex128,
        )
