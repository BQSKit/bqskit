"""This module implements the U3Gate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitarymatrix import UnitaryMatrix


class U3Gate(QubitGate):
    """The U3 single qubit gate."""

    size = 1
    num_params = 3
    qasm_name = 'u3'

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = np.sin(params[0] / 2)
        eip = np.exp(1j * params[1])
        eil = np.exp(1j * params[2])

        return UnitaryMatrix(
            [
                [cos, -eil * sin],
                [eip * sin, eip * eil * cos],
            ],
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = np.sin(params[0] / 2)
        eip = np.exp(1j * params[0])
        eil = np.exp(1j * params[1])
        deip = 1j * np.exp(1j * params[0])
        deil = 1j * np.exp(1j * params[1])

        return np.array(
            [
                [  # wrt params[0]
                    [-sin, -eil * cos],
                    [eip * cos, -eip * eil * cos],
                ],

                [  # wrt params[1]
                    [0, 0],
                    [deip * sin, deip * eil * cos],
                ],

                [  # wrt params[2]
                    [0, -deil * sin],
                    [0, eip * deil * cos],
                ],
            ], dtype=np.complex128,
        )
