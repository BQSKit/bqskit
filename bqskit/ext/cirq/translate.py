"""This module implements functions for translating to and from Cirq."""
from __future__ import annotations

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

from bqskit.ir.circuit import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def cirq_to_bqskit(cc: cirq.Circuit) -> Circuit:
    """Convert Cirq's Circuit `cc` to a BQSKit Circuit."""
    circuit = OPENQASM2Language().decode(cirq.qasm(cc))
    return circuit


def bqskit_to_cirq(circuit: Circuit) -> cirq.Circuit:
    """Convert a BQSKit Circuit to Cirq's QubitCircuit."""
    return circuit_from_qasm(OPENQASM2Language().encode(circuit))
