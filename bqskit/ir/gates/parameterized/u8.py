"""This module implements the U8Gate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class U8Gate(QutritGate, DifferentiableUnitary, CachedClass, GeneralGate):
    """The U8 single qutrit gate."""

    _num_qudits = 1
    _num_params = 8

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

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
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

        return np.array(
            [
                [  # wrt params[0]
                    [-s1 * c2 * p1, c1 * p3, -s1 * s2 * p4],
                    [
                        -c1 * c2 * c3 * p1 * p2 * m3, -s1 * \
                        c3 * p2, -c1 * s2 * c3 * p2 * m3 * p4,
                    ],
                    [
                        -c1 * c2 * s3 * p1 * m3 * p5, -s1 * \
                        s3 * p5, -c1 * s2 * s3 * m3 * p4 * p5,
                    ],
                ],

                [  # wrt params[1]
                    [-c1 * s2 * p1, 0, c1 * c2 * p4],
                    [
                        c2 * s3 * m4 * m5 + s1 * s2 * c3 * p1 * p2 * m3, 0,
                        s2 * s3 * m1 * m5 - s1 * c2 * c3 * p2 * m3 * p4,
                    ],
                    [
                        s1 * s2 * s3 * p1 * m3 * p5 - c2 * c3 * m2 * m4, 0,
                        -s2 * c3 * m1 * m2 - s1 * c2 * s3 * m3 * p4 * p5,
                    ],
                ],

                [  # wrt params[2]
                    [0, 0, 0],
                    [
                        s2 * c3 * m4 * m5 + s1 * c2 * s3 * p1 * p2 * m3,
                        -c1 * s3 * p2,
                        -c2 * c3 * m1 * m5 + s1 * s2 * s3 * p2 * m3 * p4,
                    ],
                    [
                        -s1 * c2 * c3 * p1 * m3 * p5 + s2 * s3 * m2 * m4,
                        c1 * c3 * p5,
                        -c2 * s3 * m1 * m2 - s1 * s2 * c3 * m3 * p4 * p5,
                    ],
                ],

                [  # wrt params[3]
                    [1j * c1 * c2 * p1, 0, 0],
                    [
                        -1j * s1 * c2 * c3 * p1 * p2 * m3,
                        0,
                        1j * c2 * s3 * m1 * m5,
                    ],
                    [
                        -1j * s1 * c2 * s3 * p1 * m3 * p5,
                        0, -1j * c2 * c3 * m1 * m2,
                    ],
                ],

                [  # wrt params[4]
                    [0, 0, 0],
                    [
                        -1j * s1 * c2 * c3 * p1 * p2 * m3, 1j * c1 * \
                        c3 * p2, -1j * s1 * s2 * c3 * p2 * m3 * p4,
                    ],
                    [1j * s2 * c3 * m2 * m4, 0, -1j * c2 * c3 * m1 * m2],
                ],

                [  # wrt params[5]
                    [0, 1j * s1 * p3, 0],
                    [
                        1j * s1 * c2 * c3 * p1 * p2 * m3, 0,
                        1j * s1 * s2 * c3 * p2 * m3 * p4,
                    ],
                    [
                        1j * s1 * c2 * s3 * p1 * m3 * p5, 0,
                        1j * s1 * s2 * s3 * m3 * p4 * p5,
                    ],
                ],

                [  # wrt params[6]
                    [0, 0, 1j * c1 * s2 * p4],
                    [
                        -1j * s2 * s3 * m4 * m5,
                        0,
                        -1j * s1 * s2 * c3 * p2 * m3 * p4,
                    ],
                    [
                        1j * s2 * c3 * m2 * m4,
                        0,
                        -1j * s1 * s2 * s3 * m3 * p4 * p5,
                    ],
                ],

                [  # wrt params[7]
                    [0, 0, 0],
                    [-1j * s2 * s3 * m4 * m5, 0, 1j * c2 * s3 * m1 * m5],
                    [
                        -1j * s1 * c2 * s3 * p1 * m3 * p5, 1j * c1 * \
                        s3 * p5, -1j * s1 * s2 * s3 * m3 * p4 * p5,
                    ],
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
