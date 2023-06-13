"""This module implements the IToffoliGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class IToffoliGate(ConstantGate, QubitGate):
    """
    The IToffoliGate gate, equal to an iX gate with two controls.

    The iToffoli gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & i \\\\
        0 & 0 & 0 & 0 & 0 & 0 & i & 0 \\\\
        \\end{pmatrix}

    References:
        Kim, Y., Morvan, A., Nguyen, L.B. et al. High-fidelity three-qubit
        iToffoli gate for fixed-frequency superconducting qubits. Nat. Phys.
        (2022). https://doi.org/10.1038/s41567-022-01590-3
    """

    _num_qudits = 3
    _qasm_name = 'iccx'
    _utry = UnitaryMatrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1j],
            [0, 0, 0, 0, 0, 0, 1j, 0],
        ],
    )
