"""This module implements the CUGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class CUGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary controlled rotation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\exp{i\\gamma}cos(\\frac{\\theta}{2}) & -\\exp{i(\\gamma + \\lambda)}sin(\\frac{\\theta}{2}) \\\\
        0 & 0 & \\exp{i(\\gamma + \\phi)}sin(\\frac{\\theta}{2}) & \\exp{i(\\gamma + \\phi + \\lambda)}cos(\\frac{\\theta}{2}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _num_params = 4
    _qasm_name = 'cu'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        ct = np.cos(params[0] / 2)
        st = np.sin(params[0] / 2)
        cp = np.cos(params[1])
        sp = np.sin(params[1])
        cl = np.cos(params[2])
        sl = np.sin(params[2])
        cg = np.cos(params[3])
        sg = np.sin(params[3])
        el = cl + 1j * sl
        ep = cp + 1j * sp
        eg = cg + 1j * sg

        return UnitaryMatrix(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, eg*ct, -eg*el*st],
                [0, 0, eg*ep*st, eg*ep*el*ct],
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
        cg = np.cos(params[3])
        sg = np.sin(params[3])
        el = cl + 1j * sl
        ep = cp + 1j * sp
        eg = cg + 1j * sg
        del_ = -sl + 1j * cl
        dep_ = -sp + 1j * cp
        deg_ = -sg + 1j * cg

        return np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, -0.5 * eg * st, -0.5 * eg * ct * el],
                    [0, 0, 0.5 * ct * ep * eg, -0.5 * st * el * ep * eg],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, eg * st * dep_, eg * ct * el * dep_],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, -eg * st * del_],
                    [0, 0, 0, eg * ct * ep * del_],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, deg_  * ct, -deg_ * el * st],
                    [0, 0, deg_ * ep * st, deg_ * ep * el * ct],
                ],
            ], dtype=np.complex128,
        )
