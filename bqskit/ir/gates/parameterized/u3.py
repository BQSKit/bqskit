"""This module implements the U3Gate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class U3Gate(QubitGate, DifferentiableUnitary, CachedClass):
    """
    The U3 single qubit gate.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\cos{\\frac{\\theta_0}{2}} &
        -\\exp({i\\theta_2})\\sin{\\frac{\\theta_0}{2}} \\\\
        \\exp({i\\theta_1})\\sin{\\frac{\\theta_0}{2}} &
        \\exp({i(\\theta_1 + \\theta_2)})\\cos{\\frac{\\theta_0}{2}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _num_params = 3
    _qasm_name = 'u3'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
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

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
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

    @staticmethod
    def calc_params(utry: UnitaryMatrix) -> tuple[float, float, float]:
        """
        Calculate the three parameters to a U3Gate from a given unitary.

        Args:
            utry (UnitaryMatrix): The single-qubit unitary matrix to
                calculate a U3Gate's parameters for.

        Returns:
            tuple[float, float, float]: The three parameters to a U3Gate,
                such that, `U3Gate().get_unitary` will return `utry`.

        Raises:
            ValueError: If `utry` is not a single-qubit unitary.
        """

        if utry.radixes != (2,):
            raise ValueError('Expected single-qubit unitary.')

        mag = np.linalg.det(utry.numpy) ** (-1 / 2)
        special_utry = mag * utry
        a = np.angle(special_utry[1, 1])
        b = np.angle(special_utry[1, 0])
        c = np.abs(special_utry[1, 0])
        d = np.abs(special_utry[0, 0])
        theta = 2 * float(np.arctan2(c, d))
        phi = (a + b)
        lamb = (a - b)
        return theta, phi, lamb
