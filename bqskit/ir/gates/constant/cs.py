"""This module implements the CSGate."""

from __future__ import annotations

from openqudit.expressions import Controlled as _Controlled
from openqudit.expressions import SGate as _SGate

from bqskit.ir.gate import Gate
from bqskit.utils.cachedclass import CachedClass


class CSGate(Gate, CachedClass):
    """
    The Controlled-S gate.

    The CS gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'cs'
    _expr = _Controlled(_SGate())

    def get_qasm_gate_def(self) -> str:
        """
        Returns a qasm gate definition block for this gate.

        Note:
            `cs` is not part of the standard `qelib1.inc` gate library,
            unlike most other two-qubit gates BQSKit exports. BQSKit's own
            parser recognizes `cs` natively regardless, so this definition
            is only emitted for interoperability with other QASM consumers
            (e.g. Qiskit) that would otherwise reject it as undefined. See
            https://github.com/BQSKit/bqskit/issues/306.
        """
        return 'gate cs a,b { cu1(pi/2) a,b; }'
