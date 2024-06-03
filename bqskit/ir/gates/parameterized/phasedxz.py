"""This module implements the PhasedXZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class PhasedXZGate(QubitGate, DifferentiableUnitary, CachedClass):
    """
    A gate representing an Google's PhasedXZGate.

    References:
        https://quantumai.google/reference/python/cirq/PhasedXZGate
    """

    _num_qudits = 1
    _num_params = 3
    _qasm_name = 'pxz'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        x = params[0]
        z = params[1]
        a = params[2]
        cos = np.cos(np.pi * x / 2)
        sin = -1j * np.sin(np.pi * x / 2)
        e1 = np.exp(1j * np.pi * x / 2)
        e2 = np.exp(1j * np.pi * (x / 2 - a))
        e3 = np.exp(1j * np.pi * (x / 2 + z + a))
        e4 = np.exp(1j * np.pi * (x / 2 + z))

        return UnitaryMatrix(
            [
                [e1 * cos, e2 * sin],
                [e3 * sin, e4 * cos],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        x = params[0]
        z = params[1]
        a = params[2]
        cos = np.cos(np.pi * x / 2)
        sin = -1j * np.sin(np.pi * x / 2)
        dcos = -np.pi * np.sin(np.pi * x / 2) / 2
        dsin = -1j * np.pi * np.cos(np.pi * x / 2) / 2
        e1 = np.exp(1j * np.pi * x / 2)
        e2 = np.exp(1j * np.pi * (x / 2 - a))
        e3 = np.exp(1j * np.pi * (x / 2 + z + a))
        e4 = np.exp(1j * np.pi * (x / 2 + z))

        return np.array(
            [
                [
                    [
                        (e1 * dcos) + (1j * np.pi / 2 * e1 * cos),
                        (e2 * dsin) + (1j * np.pi / 2 * e2 * sin),
                    ],
                    [
                        (e3 * dsin) + (1j * np.pi / 2 * e3 * sin),
                        (e4 * dcos) + (1j * np.pi / 2 * e4 * cos),
                    ],
                ],
                [
                    [0, 0],
                    [1j * np.pi * e3 * sin, 1j * np.pi * e4 * cos],
                ],
                [
                    [0, -1j * np.pi * e2 * sin],
                    [1j * np.pi * e3 * sin, 0],
                ],
            ], dtype=np.complex128,
        )

    def get_qasm_gate_def(self) -> str:
        """Return the QASM gate definition for this gate."""
        theta0 = 'x*pi'
        theta1 = '(z+a-0.5)*pi'
        theta2 = '(0.5-a)*pi'
        return (f'gate {self._qasm_name} a,b,c'
                f' {{ u3({theta0},{theta1},{theta2}); }}')

    def get_inverse_params(self, params: RealVector = []) -> RealVector:
        """Return the inverse parameters for this gate."""
        self.check_parameters(params)
        return [-params[0], -params[1], params[1] + params[2]]

    def get_inverse(self) -> Gate:
        """Return the inverse of this gate."""
        return PhasedXZGate()
