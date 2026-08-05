"""This module implements the IToffoliGate."""

from __future__ import annotations

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import UnitaryExpression as _UnitaryExpression

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class IToffoliGate(Gate, CachedClass):
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
    _expr = _Controlled(
        _UnitaryExpression('iX() { [[0,i],[i,0]] }'),
        [2, 2],
    )
