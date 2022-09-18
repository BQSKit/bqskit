"""This module implements pre-built models for Google's QPUs."""
from __future__ import annotations

import cirq_google

from bqskit.compiler.machine import MachineModel
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CZGate
from bqskit.ir.gates import PhasedXZGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SqrtISwapGate
from bqskit.ir.gates import SycamoreGate
from bqskit.qis.graph import CouplingGraph

google_gate_set: set[Gate] = {
    PhasedXZGate(),
    RZGate(),
    SycamoreGate(),
    CZGate(),
    SqrtISwapGate(),
}

# Sycamore Device
_qubits = cirq_google.Sycamore.metadata.qubit_set
_qubit_map = {q: i for i, q in enumerate(list(_qubits))}
_pairs = cirq_google.Sycamore.metadata.qubit_pairs
_edges = [(_qubit_map[q1], _qubit_map[q2]) for q1, q2 in _pairs]
_sycamore_coupling_graph = CouplingGraph(_edges)
SycamoreModel = MachineModel(54, _sycamore_coupling_graph, google_gate_set)

# Sycamore23 Device
_qubits = cirq_google.Sycamore23.metadata.qubit_set
_qubit_map = {q: i for i, q in enumerate(list(_qubits))}
_pairs = cirq_google.Sycamore23.metadata.qubit_pairs
_edges = [(_qubit_map[q1], _qubit_map[q2]) for q1, q2 in _pairs]
_sycamore23_coupling_graph = CouplingGraph(_edges)
Sycamore23Model = MachineModel(23, _sycamore23_coupling_graph, google_gate_set)

__all__ = ['SycamoreModel', 'Sycamore23Model']
