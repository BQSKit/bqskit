"""This module implements functions for translating to and from QuTiP."""
from __future__ import annotations

from qutip import QubitCircuit
from qutip.qip.qasm import circuit_to_qasm_str
from qutip.qip.qasm import read_qasm

from bqskit.ir.circuit import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def qutip_to_bqskit(qc: QubitCircuit) -> Circuit:
    """Convert QuTiP's QubitCircuit `qc` to a BQSKit Circuit."""
    circuit = OPENQASM2Language().decode(circuit_to_qasm_str(qc))
    return circuit


def bqskit_to_qutip(circuit: Circuit) -> QubitCircuit:
    """Convert a BQSKit Circuit to QuTiP's QubitCircuit."""
    return read_qasm(OPENQASM2Language().encode(circuit), strmode=True)
