# flake8: noqa
from __future__ import annotations

import pytest
pytest.importorskip('pytket')
pytest.importorskip('pytket.extensions.qiskit')

from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.circuit import Circuit
from bqskit.ext import pytket_to_bqskit
from bqskit.ext import bqskit_to_pytket
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.compile import compile
# from pytket.extensions.qiskit import AerUnitaryBackend
from pytket.circuit import Circuit as QubitCircuit
from pytket import OpType
import numpy as np


class TestTranslate:
    @pytest.fixture
    def bqskit_circuit(self) -> Circuit:
        circuit = Circuit(3)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(U3Gate(), 0, [1, 2, 3])
        circuit.append_gate(U3Gate(), 1, [1, 2, 3])
        circuit.append_gate(U3Gate(), 2, [1, 2, 3])
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(U3Gate(), 0, [1, 2.4, 3])
        circuit.append_gate(U3Gate(), 1, [1, 2.2, 3])
        circuit.append_gate(U3Gate(), 2, [1, 2.1, 3])
        circuit.append_gate(U3Gate(), 2, [1, 2.1, 3])
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 2))
        circuit.append_gate(U3Gate(), 0, [1, 2.4, 3])
        circuit.append_gate(U3Gate(), 1, [1, 2.2, 3])
        circuit.append_gate(U3Gate(), 2, [1, 2.1, 3])
        return circuit

    @pytest.fixture
    def pytket_circuit(self) -> QubitCircuit:
        circuit = QubitCircuit(3)
        circuit.CX(0, 1)
        circuit.add_gate(OpType.U3, [1, 2, 3], [0])
        circuit.add_gate(OpType.U3, [1, 2, 3], [1])
        circuit.add_gate(OpType.U3, [1, 2, 3], [2])
        circuit.CX(0, 1)
        circuit.CX(0, 2)
        circuit.CX(0, 2)
        circuit.add_gate(OpType.U3, [1, 2.4, 3], [0])
        circuit.add_gate(OpType.U3, [1, 2.2, 3], [1])
        circuit.add_gate(OpType.U3, [1, 2.1, 3], [2])
        circuit.add_gate(OpType.U3, [1, 2.1, 3], [2])
        circuit.CX(0, 2)
        circuit.CX(0, 2)
        circuit.CX(0, 1)
        circuit.CX(0, 2)
        circuit.add_gate(OpType.U3, [1, 2.4, 3], [0])
        circuit.add_gate(OpType.U3, [1, 2.2, 3], [1])
        circuit.add_gate(OpType.U3, [1, 2.1, 3], [2])
        return circuit

    def get_unitary(self, qc: QubitCircuit) -> np.ndarray:  # type: ignore
        return qc.get_unitary()

    def test_bqskit_to_bqskit(self, bqskit_circuit: Circuit) -> None:
        in_utry = bqskit_circuit.get_unitary()
        out_circuit = pytket_to_bqskit(bqskit_to_pytket(bqskit_circuit))
        out_utry = out_circuit.get_unitary()
        assert in_utry.get_distance_from(out_utry) < 1e-7

    def test_pytket_to_pytket(self, pytket_circuit: QubitCircuit) -> None:
        in_utry = UnitaryMatrix(self.get_unitary(pytket_circuit))
        out_circuit = bqskit_to_pytket(pytket_to_bqskit(pytket_circuit))
        out_utry = UnitaryMatrix(self.get_unitary(out_circuit))
        assert in_utry.get_distance_from(out_utry) < 1e-7

    def test_compile_bqskit(
        self, pytket_circuit: QubitCircuit,
        compiler:Compiler,
    ) -> None:
        in_utry = UnitaryMatrix(self.get_unitary(pytket_circuit))
        bqskit_circuit = pytket_to_bqskit(pytket_circuit)
        bqskit_out_circuit = compile(bqskit_circuit, max_synthesis_size=2, compiler=compiler)
        out_circuit = bqskit_to_pytket(bqskit_out_circuit)
        out_utry = UnitaryMatrix(self.get_unitary(out_circuit))
        assert in_utry.get_distance_from(out_utry) < 1e-5

    def test_synthesis_bqskit(
        self, pytket_circuit: QubitCircuit,
        compiler:Compiler,
    ) -> None:
        in_utry = UnitaryMatrix(self.get_unitary(pytket_circuit))
        bqskit_circuit = pytket_to_bqskit(pytket_circuit)
        bqskit_out_circuit = compile(bqskit_circuit, compiler=compiler)
        out_circuit = bqskit_to_pytket(bqskit_out_circuit)
        out_utry = UnitaryMatrix(self.get_unitary(out_circuit))
        assert in_utry.get_distance_from(out_utry) < 1e-5
