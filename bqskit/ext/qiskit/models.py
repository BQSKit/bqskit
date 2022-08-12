"""This module implements functions for working with IBM Backends."""
from __future__ import annotations

from qiskit.providers import BackendV1

from bqskit.compiler import MachineModel
from bqskit.ir.gate import Gate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.sx import SXGate
from bqskit.ir.gates.parameterized import RZGate
from bqskit.ir.gates.parameterized.u1 import U1Gate
from bqskit.ir.gates.parameterized.u2 import U2Gate
from bqskit.ir.gates.parameterized.u3 import U3Gate


def model_from_backend(backend: BackendV1) -> MachineModel:
    """Create a machine model for a IBM Backend."""
    config = backend.configuration()
    num_qudits = config['n_qubits']
    gate_set = _basis_gate_str_to_bqskit_gate(config['basis_gates'])
    coupling_map = config['coupling_map']
    return MachineModel(num_qudits, coupling_map, gate_set)


def _basis_gate_str_to_bqskit_gate(basis_gates: list[str]) -> set[Gate]:
    gate_set: set[Gate] = set()
    for basis_gate in basis_gates:
        if basis_gate == 'cx':
            gate_set.add(CNOTGate())
        if basis_gate == 'cz':
            gate_set.add(CZGate())
        elif basis_gate == 'u3':
            gate_set.add(U3Gate())
        elif basis_gate == 'u2':
            gate_set.add(U2Gate())
        elif basis_gate == 'u1':
            gate_set.add(U1Gate())
        elif basis_gate == 'sx':
            gate_set.add(SXGate())
        elif basis_gate == 'p':
            gate_set.add(RZGate())
    return gate_set
