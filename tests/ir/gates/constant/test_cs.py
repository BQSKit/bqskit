from __future__ import annotations

import pytest

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CSGate
from bqskit.ir.lang.qasm2 import OPENQASM2Language


def test_cs_encode_decode() -> None:
    c = Circuit(2)
    c.append_gate(CSGate(), (0, 1))

    output_qasm = c.to('qasm')

    resulting_circuit = OPENQASM2Language().decode(output_qasm)
    assert resulting_circuit[0, 0].gate == CSGate()


def test_cs_qasm_includes_gate_definition() -> None:
    # `cs` is not part of the standard qelib1.inc library, unlike most other
    # constant two-qubit gates -- BQSKit must emit its own definition block
    # for other QASM consumers to recognize it. See issue #306.
    c = Circuit(2)
    c.append_gate(CSGate(), (0, 1))

    output_qasm = c.to('qasm')

    assert 'gate cs' in output_qasm


def test_cs_qasm_importable_by_qiskit() -> None:
    # Regression test for #306: a `cs`-containing circuit exported by BQSKit
    # used to fail re-import in other QASM consumers (e.g. Qiskit) with
    # `QASM2ParseError: 'cs' is not defined in this scope`, because BQSKit
    # emitted no definition for it while assuming it lives in qelib1.inc.
    pytest.importorskip('qiskit')
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    c = Circuit(2)
    c.append_gate(CSGate(), (0, 1))
    output_qasm = c.to('qasm')

    qc = QuantumCircuit.from_qasm_str(output_qasm)

    bqskit_utry = c.get_unitary()
    qiskit_utry = Operator(qc).data
    assert bqskit_utry.get_distance_from(qiskit_utry) < 1e-7
