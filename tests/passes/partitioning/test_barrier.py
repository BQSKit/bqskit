from __future__ import annotations

import numpy as np

from bqskit import compile
from bqskit.compiler.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import BarrierPlaceholder
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import U3Gate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.passes.partitioning.quick import QuickPartitioner


def test_barrier_still_there_after_partitioning(compiler: Compiler) -> None:
    circuit = Circuit(2)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(BarrierPlaceholder(2), (0, 1))
    circuit.append_gate(CXGate(), (0, 1))
    circuit = compiler.compile(circuit, [QuickPartitioner(3)])
    assert circuit.num_operations == 3
    assert isinstance(circuit[0, 0].gate, CircuitGate)
    assert isinstance(circuit[1, 0].gate, BarrierPlaceholder)
    assert isinstance(circuit[2, 0].gate, CircuitGate)
    circuit.unfold_all()
    assert isinstance(circuit[0, 0].gate, CXGate)
    assert isinstance(circuit[1, 0].gate, BarrierPlaceholder)
    assert isinstance(circuit[2, 0].gate, CXGate)


def test_barrier_stop_partitioning_across_some_circuit(
        compiler: Compiler,
) -> None:
    circuit = Circuit(4)
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(BarrierPlaceholder(2), (0, 1))
    circuit.append_gate(CXGate(), (0, 1))
    circuit.append_gate(CXGate(), (2, 3))
    circuit.append_gate(CXGate(), (2, 3))
    circuit = compiler.compile(circuit, [QuickPartitioner(2)])
    assert circuit.num_operations == 4


def test_barrier_end_to_end_single_qubit_syn(compiler: Compiler) -> None:
    circuit = Circuit(1)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(BarrierPlaceholder(1), 0)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(BarrierPlaceholder(1), 0)
    circuit.append_gate(U3Gate(), 0)
    circuit.append_gate(BarrierPlaceholder(1), 0)
    circuit.append_gate(U3Gate(), 0)
    out = compile(circuit, compiler=compiler)
    assert out.num_operations == 7


def test_barrier_corner_case(compiler: Compiler) -> None:
    input = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        barrier q[0],q[1];
        h q[0];
        barrier q[0],q[1];
        h q[0];
    """

    circuit = OPENQASM2Language().decode(input)
    out_circuit = compiler.compile(circuit, [QuickPartitioner(3)])
    assert out_circuit.get_unitary().get_distance_from(np.eye(4)) < 1e-8


def test_barrier_corner_case_2(compiler: Compiler) -> None:
    input = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        barrier q[0],q[1],q[2];
        cx q[0],q[1];
        cx q[1],q[2];
        cx q[1],q[2];
        cx q[0],q[1];
    """

    circuit = OPENQASM2Language().decode(input)
    out_circuit = compiler.compile(circuit, [QuickPartitioner(3)])
    assert out_circuit.get_unitary().get_distance_from(np.eye(8)) < 1e-8


def test_barrier_corner_case_3(compiler: Compiler) -> None:
    input = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        cx q[0],q[1];
        cx q[0],q[1];
        barrier q;
    """

    circuit = OPENQASM2Language().decode(input)
    out_circuit = compiler.compile(circuit, [QuickPartitioner(3)])
    assert out_circuit.get_unitary().get_distance_from(np.eye(8)) < 1e-8


def test_barrier_corner_case_4(compiler: Compiler) -> None:
    input = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        h q[1];
        cx q[2],q[3];
        cx q[0],q[2];
        barrier q[0], q[1];
    """

    circuit = OPENQASM2Language().decode(input)
    out_circuit = compiler.compile(circuit, [QuickPartitioner(3)])
    correct_circuit = Circuit(4)
    correct_circuit.append_gate(HGate(), 1)
    correct_circuit.append_gate(CXGate(), (2, 3))
    correct_circuit.append_gate(CXGate(), (0, 2))
    correct_utry = correct_circuit.get_unitary()
    assert out_circuit.get_unitary().get_distance_from(correct_utry) < 1e-7



def test_barrier_corner_case_5(compiler: Compiler) -> None:
    input = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        cx q[0],q[1];
        barrier q[1], q[2];
        cx q[0],q[2];
    """

    circuit = OPENQASM2Language().decode(input)
    out_circuit = compiler.compile(circuit, [QuickPartitioner(3)])
    correct_circuit = Circuit(3)
    correct_circuit.append_gate(CXGate(), (0, 1))
    correct_circuit.append_gate(CXGate(), (0, 2))
    correct_utry = correct_circuit.get_unitary()
    assert out_circuit.get_unitary().get_distance_from(correct_utry) < 1e-7
