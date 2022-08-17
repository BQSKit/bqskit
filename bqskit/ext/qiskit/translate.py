"""This module implements functions for translating to and from Qiskit."""
from __future__ import annotations

from qiskit import QuantumCircuit

from bqskit.ir.circuit import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def qiskit_to_bqskit(qc: QuantumCircuit) -> Circuit:
    """Convert Qiskit's QuantumCircuit `qc` to a BQSKit Circuit."""
    circuit = OPENQASM2Language().decode(qc.qasm())
    # circuit.renumber_qudits(list(reversed(range(circuit.num_qudits))))
    return circuit
    # TODO: support gates not captured by qasm


def bqskit_to_qiskit(circuit: Circuit) -> QuantumCircuit:
    """Convert a BQSKit Circuit to Qiskit's QuantumCircuit."""
    # circuit.renumber_qudits(list(reversed(range(circuit.num_qudits))))
    return QuantumCircuit.from_qasm_str(OPENQASM2Language().encode(circuit))
    # TODO: support gates not captured by qasm
