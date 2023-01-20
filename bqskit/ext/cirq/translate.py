"""This module implements functions for translating to and from Cirq."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import cirq

from bqskit.ir.circuit import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def cirq_to_bqskit(cc: cirq.Circuit) -> Circuit:
    """Convert Cirq's Circuit `cc` to a BQSKit Circuit."""
    try:
        import cirq
    except ImportError as e:
        raise ImportError(
            '\n\nUnable to import cirq package.\n'
            'Please ensure that it is installed with the following command:\n'
            '\tpip install cirq\n',
        ) from e

    circuit = OPENQASM2Language().decode(cirq.qasm(cc))
    return circuit


def bqskit_to_cirq(circuit: Circuit) -> cirq.Circuit:
    """Convert a BQSKit Circuit to Cirq's Circuit."""
    try:
        from cirq.contrib.qasm_import import circuit_from_qasm
    except ImportError as e:
        raise ImportError(
            '\n\nUnable to import cirq package.\n'
            'Please ensure that it is installed with the following command:\n'
            '\tpip install cirq\n',
        ) from e
        
    return circuit_from_qasm(OPENQASM2Language().encode(circuit))
