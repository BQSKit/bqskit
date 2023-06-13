from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CKMDGate(QutritGate, DifferentiableUnitary, CachedClass):
    """The Cabibbo–Kobayashi–Maskawa dagger single qutrit gate."""

    _num_qudits = 1
    _num_params = 4
    _qasm_name = 'CKMD'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        s1 = np.sin(-params[0])
        c1 = np.cos(-params[0])
        s2 = np.sin(-params[1])
        c2 = np.cos(-params[1])
        s3 = np.sin(-params[2])
        c3 = np.cos(-params[2])

        p1 = np.exp(-1j * params[3])
        m1 = np.exp(1j * params[3])

        u1 = np.array([[1, 0, 0], [0, c3, s3], [0, -s3, c3]])
        u2 = np.array([[c1, 0, s1 * m1], [0, 1, 0], [-s1 * p1, 0, c1]])
        u3 = np.array([[c2, s2, 0], [-s2, c2, 0], [0, 0, 1]])

        return UnitaryMatrix(u1 @ u2 @ u3)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        s1 = np.sin(-params[0])
        c1 = np.cos(-params[0])
        s2 = np.sin(-params[1])
        c2 = np.cos(-params[1])
        s3 = np.sin(-params[2])
        c3 = np.cos(-params[2])

        p1 = np.exp(-1j * params[3])
        m1 = np.exp(1j * params[3])

        u1 = np.array([[1, 0, 0], [0, c3, s3], [0, -s3, c3]])
        u2 = np.array([[c1, 0, s1 * m1], [0, 1, 0], [-s1 * p1, 0, c1]])
        u3 = np.array([[c2, s2, 0], [-s2, c2, 0], [0, 0, 1]])

        u1p = np.array([[0, 0, 0], [0, -s3, c3], [0, -c3, -s3]])
        u2p1 = np.array([
            [-s1, 0, c1 * m1], [0, 0, 0],
            [-c1 * m1, 0, s1],
        ])
        u2p2 = np.array([
            [0, 0, -1j * s1 * m1], [0, 0, 0],
            [-1j * s1 * p1, 0, 0],
        ])
        u3p = np.array([[s2, c2, 0], [-c2, -s2, 0], [0, 0, 0]])

        return np.array(
            [
                # wrt params[0] -> 1-3
                u1 @ u2p1 @ u3,

                # wrt params[1] -> 1-2
                u1 @ u2 @ u3p,

                # wrt params[2] -> 2-3
                u1p @ u2 @ u3,
                # wrt params[3] -> cp
                u1 @ u2p2 @ u3,
            ], dtype=np.complex128,
        )
