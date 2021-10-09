"""This module implements the U3Gate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class U3Gate(QubitGate, DifferentiableUnitary, CachedClass):
    """The U3 single qubit gate."""

    _num_qudits = 1
    _num_params = 3
    _qasm_name = 'u3'

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        ct = np.cos(params[0] / 2)
        st = np.sin(params[0] / 2)
        cp = np.cos(params[1])
        sp = np.sin(params[1])
        cl = np.cos(params[2])
        sl = np.sin(params[2])
        el = cl + 1j * sl
        ep = cp + 1j * sp

        return UnitaryMatrix(
            [
                [ct, -el * st],
                [ep * st, ep * el * ct],
            ],
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        ct = np.cos(params[0] / 2)
        st = np.sin(params[0] / 2)
        cp = np.cos(params[1])
        sp = np.sin(params[1])
        cl = np.cos(params[2])
        sl = np.sin(params[2])
        el = cl + 1j * sl
        ep = cp + 1j * sp
        del_ = -sl + 1j * cl
        dep_ = -sp + 1j * cp

        return np.array(
            [
                [  # wrt params[0]
                    [-0.5 * st, -0.5 * ct * el],
                    [0.5 * ct * ep, -0.5 * st * el * ep],
                ],

                [  # wrt params[1]
                    [0, 0],
                    [st * dep_, ct * el * dep_],
                ],

                [  # wrt params[2]
                    [0, -st * del_],
                    [0, ct * ep * del_],
                ],
            ], dtype=np.complex128,
        )
