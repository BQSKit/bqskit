"""This module implements functions for translating to and from PyTKet."""
from __future__ import annotations

import pytket
from pytket.qasm import circuit_from_qasm_str
from pytket.qasm import circuit_to_qasm_str

from bqskit.ir.circuit import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def pytket_to_bqskit(qc: pytket.Circuit) -> Circuit:
    """Convert PyTKet's Circuit `cc` to a BQSKit Circuit."""
    circuit = OPENQASM2Language().decode(circuit_to_qasm_str(qc))
    return circuit


def bqskit_to_pytket(circuit: Circuit) -> pytket.Circuit:
    """Convert a BQSKit Circuit to PyTKet's QubitCircuit."""
    return circuit_from_qasm_str(OPENQASM2Language().encode(circuit))
