"""This module implements the rotations due to the generatos of SU(3)."""
from __future__ import annotations

import numpy as np
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.typing import is_integer

_SU3_GENERATOR_QGL = {
    0: 'RSU3_0(t0) { [[cos(t0),~i*sin(t0),0],[~i*sin(t0),cos(t0),0],[0,0,1]] }',
    1: 'RSU3_1(t0) { [[cos(t0),~sin(t0),0],[sin(t0),cos(t0),0],[0,0,1]] }',
    2: 'RSU3_2(t0) { [[e^(~i*t0),0,0],[0,e^(i*t0),0],[0,0,1]] }',
    3: 'RSU3_3(t0) { [[cos(t0),0,~i*sin(t0)],[0,1,0],[~i*sin(t0),0,cos(t0)]] }',
    4: 'RSU3_4(t0) { [[cos(t0),0,~sin(t0)],[0,1,0],[sin(t0),0,cos(t0)]] }',
    5: 'RSU3_5(t0) { [[1,0,0],[0,cos(t0),~i*sin(t0)],[0,~i*sin(t0),cos(t0)]] }',
    6: 'RSU3_6(t0) { [[1,0,0],[0,cos(t0),~sin(t0)],[0,sin(t0),cos(t0)]] }',
    7: (
        'RSU3_7(t0) { [[e^(~i*t0/sqrt(3)),0,0],'
        '[0,e^(~i*t0/sqrt(3)),0],[0,0,e^(2*i*t0/sqrt(3))]] }'
    ),
}


class RSU3Gate(Gate, CachedClass):
    """
    Rotation by SU3 generator for a single qutrit gate.

    .. math::
        \\exp(i * \\theta * \\lambda_j)

    where lambda_j is the j-th generator of the SU(3) Lie algebra.
    We use the physics notation, so each generator is Hermitian and not
    necessarily unitary. See See
    :func:`~bqskit.utils.math.compute_su_generators` for more info on the
    generators. There are N^2-1 = 3^2-1 = 8 generators and N-1 = 3-1 = 2
    mutually commute.

    References:
        - "Lie algebras in particle physics : from isospin to unified theories",
          Howard Georgi
    """

    _num_qudits = 1
    _num_params = 1

    def __init__(self, index: int) -> None:
        """
        Construct a RSU3Gate.

        Args:
            index (int): The index of the generator to use. See
                :func:`~bqskit.utils.math.compute_su_generators` for more
                info on the generators.

        Raises:
            ValueError: If index is less than 0 or greater than 7
        """
        if not is_integer(index):
            raise TypeError('RSU3Gate generator index must be an integer.')

        if index < 0 or index > 7:
            raise ValueError(f'Expected index between 0 and 7, got {index}.')

        self.index = index
        self._expr = _UnitaryExpression(_SU3_GENERATOR_QGL[index])

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""

        self.check_parameters(params)

        cos = np.cos(params[0])
        sin = np.sin(params[0])

        matrix = np.eye(3, dtype=np.complex128)

        if self.index == 0:
            matrix[0, 0] = cos
            matrix[1, 1] = cos
            matrix[0, 1] = -1j * sin
            matrix[1, 0] = -1j * sin
        elif self.index == 1:
            matrix[0, 0] = cos
            matrix[1, 1] = cos
            matrix[0, 1] = -sin
            matrix[1, 0] = sin
        elif self.index == 2:
            matrix[0, 0] = np.exp(-1j * params[0])
            matrix[1, 1] = np.exp(1j * params[0])
        elif self.index == 3:
            matrix[0, 0] = cos
            matrix[2, 2] = cos
            matrix[0, 2] = -1j * sin
            matrix[2, 0] = -1j * sin
        elif self.index == 4:
            matrix[0, 0] = cos
            matrix[2, 2] = cos
            matrix[0, 2] = -sin
            matrix[2, 0] = sin
        elif self.index == 5:
            matrix[1, 1] = cos
            matrix[2, 2] = cos
            matrix[1, 2] = -1j * sin
            matrix[2, 1] = -1j * sin
        elif self.index == 6:
            matrix[1, 1] = cos
            matrix[2, 2] = cos
            matrix[1, 2] = -sin
            matrix[2, 1] = sin
        elif self.index == 7:
            matrix[0, 0] = np.exp(-1j * params[0] / np.sqrt(3))
            matrix[1, 1] = np.exp(-1j * params[0] / np.sqrt(3))
            matrix[2, 2] = np.exp(2j * params[0] / np.sqrt(3))
        return UnitaryMatrix(matrix, self.radixes)
