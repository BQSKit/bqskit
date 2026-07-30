"""This module implements the U8Gate."""
from __future__ import annotations

import numpy as np
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gates import GeneralGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class U8Gate(GeneralGate, CachedClass):
    """The U8 single qutrit gate."""

    _num_qudits = 1
    _num_params = 8
    _expr = _UnitaryExpression(
        'U8(t0,t1,t2,t3,t4,t5,t6,t7) { ['
        '[cos(t0)*cos(t1)*e^(i*t3), sin(t0)*e^(i*t5), '
        'cos(t0)*sin(t1)*e^(i*t6)],'
        '[sin(t1)*sin(t2)*e^(~i*t6)*e^(~i*t7) - '
        'sin(t0)*cos(t1)*cos(t2)*e^(i*t3)*e^(i*t4)*e^(~i*t5), '
        'cos(t0)*cos(t2)*e^(i*t4), '
        '~cos(t1)*sin(t2)*e^(~i*t3)*e^(~i*t7) - '
        'sin(t0)*sin(t1)*cos(t2)*e^(i*t4)*e^(~i*t5)*e^(i*t6)],'
        '[~sin(t0)*cos(t1)*sin(t2)*e^(i*t3)*e^(~i*t5)*e^(i*t7) - '
        'sin(t1)*cos(t2)*e^(~i*t4)*e^(~i*t6), '
        'cos(t0)*sin(t2)*e^(i*t7), '
        'cos(t1)*cos(t2)*e^(~i*t3)*e^(~i*t4) - '
        'sin(t0)*sin(t1)*sin(t2)*e^(~i*t5)*e^(i*t6)*e^(i*t7)]'
        '] }',
    )

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        s1 = np.sin(params[0])
        c1 = np.cos(params[0])
        s2 = np.sin(params[1])
        c2 = np.cos(params[1])
        s3 = np.sin(params[2])
        c3 = np.cos(params[2])

        p1 = np.exp(1j * params[3])
        m1 = np.exp(-1j * params[3])
        p2 = np.exp(1j * params[4])
        m2 = np.exp(-1j * params[4])
        p3 = np.exp(1j * params[5])
        m3 = np.exp(-1j * params[5])
        p4 = np.exp(1j * params[6])
        m4 = np.exp(-1j * params[6])
        p5 = np.exp(1j * params[7])
        m5 = np.exp(-1j * params[7])

        return UnitaryMatrix(
            [
                [
                    c1 * c2 * p1,
                    s1 * p3,
                    c1 * s2 * p4,
                ],
                [
                    s2 * s3 * m4 * m5 - s1 * c2 * c3 * p1 * p2 * m3,
                    c1 * c3 * p2,
                    -c2 * s3 * m1 * m5 - s1 * s2 * c3 * p2 * m3 * p4,
                ],
                [
                    -s1 * c2 * s3 * p1 * m3 * p5 - s2 * c3 * m2 * m4,
                    c1 * s3 * p5,
                    c2 * c3 * m1 * m2 - s1 * s2 * s3 * m3 * p4 * p5,
                ],
            ],
        )

    def calc_params(self, utry: UnitaryMatrix) -> list[float]:
        """Return the parameters for this gate to implement `utry`"""
        if utry.radixes != (3,):
            raise ValueError('Expected single-qutrit unitary.')

        params = [0.0] * 8

        mag = np.linalg.det(utry.numpy) ** (-1 / 3)
        su = mag * utry

        params[5] = np.arctan2(su[0][1].imag, su[0][1].real)

        s1 = (su[0][1] * np.exp(-1j * params[5])).real
        params[0] = np.arcsin(s1)

        c2p1 = su[0][0] / np.cos(params[0])
        params[3] = np.arctan2(c2p1.imag, c2p1.real)

        c2 = (c2p1 * np.exp(-1j * params[3])).real
        params[1] = np.arccos(c2)

        c3p2 = su[1][1] / np.cos(params[0])
        params[4] = np.arctan2(c3p2.imag, c3p2.real)

        c3 = (c3p2 * np.exp(-1j * params[4])).real
        params[2] = np.arccos(c3)

        p4 = su[0][2] / (np.cos(params[0]) * np.sin(params[1]))
        params[6] = np.arctan2(p4.imag, p4.real)

        p5 = su[2][1] / (np.cos(params[0]) * np.sin(params[2]))
        params[7] = np.arctan2(p5.imag, p5.real)

        return params
