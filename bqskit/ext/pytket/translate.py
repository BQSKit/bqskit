"""This module implements functions for translating to and from PyTKet."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pytket

from bqskit.ir.circuit import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def pytket_to_bqskit(qc: pytket.Circuit) -> Circuit:
    """Convert PyTKet's Circuit `cc` to a BQSKit Circuit."""
    try:
        from pytket.qasm import circuit_to_qasm_str
    except ImportError as e:
        raise ImportError(
            '\n\nUnable to import pytket package.\n'
            'Please ensure that it is installed with the following command:\n'
            '\tpip install pytket\n',
        ) from e

    circuit = OPENQASM2Language().decode(circuit_to_qasm_str(qc))
    return circuit


def bqskit_to_pytket(circuit: Circuit) -> pytket.Circuit:
    """Convert a BQSKit Circuit to a PyTKet Circuit."""
    try:
        from pytket.qasm import circuit_from_qasm_str
    except ImportError as e:
        raise ImportError(
            '\n\nUnable to import pytket package.\n'
            'Please ensure that it is installed with the following command:\n'
            '\tpip install pytket\n',
        ) from e

    return circuit_from_qasm_str(OPENQASM2Language().encode(circuit))
