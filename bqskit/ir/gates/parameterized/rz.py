"""This module implements the RZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from openqudit.expressions import RZGate as _RZGate

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.utils.cachedclass import CachedClass


class RZGate(Gate, CachedClass):
    """
    A gate representing an arbitrary rotation around the Z axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\exp({-i\\frac{\\theta}{2}}) & 0 \\\\
        0 & \\exp({i\\frac{\\theta}{2}}) \\\\
        \\end{pmatrix}
    """

    _qasm_name = 'rz'
    _expr = _RZGate()

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`~bqskit.ir.gate.Gate` for more info.
        """
        self.check_parameters(params)

        dpexp = 1j * np.exp(1j * params[0] / 2) / 2
        dnexp = -1j * np.exp(-1j * params[0] / 2) / 2

        return np.array(
            [
                [
                    [dnexp, 0],
                    [0, dpexp],
                ],
            ], dtype=np.complex128,
        )
