"""This module implements the PhasedXZGate."""

from __future__ import annotations

import numpy as np
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class PhasedXZGate(Gate, CachedClass):
    """
    A gate representing an Google's PhasedXZGate.

    References:
        https://quantumai.google/reference/python/cirq/PhasedXZGate
    """

    _num_qudits = 1
    _num_params = 3
    _qasm_name = 'pxz'
    _expr = _UnitaryExpression(
        'PhasedXZ(t0,t1,t2) { '
        '[[e^(i*pi*t0/2)*cos(pi*t0/2),'
        ' e^(i*pi*(t0/2-t2))*(~i*sin(pi*t0/2))],'
        '[e^(i*pi*(t0/2+t1+t2))*(~i*sin(pi*t0/2)),'
        ' e^(i*pi*(t0/2+t1))*cos(pi*t0/2)]] }',
    )

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
