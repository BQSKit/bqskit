"""This file tests the qasm2 language support in BQSKit."""
from __future__ import annotations

from unittest.mock import mock_open
from unittest.mock import patch

import pytest

from numpy.testing import assert_almost_equal

from bqskit.ext.qiskit import qiskit_to_bqskit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.ecr import ECRGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.gates.parameterized.u1 import U1Gate
from bqskit.ir.gates.parameterized.u1q import U1qGate
from bqskit.ir.gates.parameterized.u2 import U2Gate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.reset import Reset
from bqskit.ir.lang.language import LangException
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TestComment:
    def test_comment_header_1(self) -> None:
        input = """
            // This is a comment
            OPENQASM 2.0;
            qreg q[1];
            u1(0.1) q[0];
        """
        circuit = OPENQASM2Language().decode(input)
        assert circuit.get_unitary() == U1Gate().get_unitary([0.1])

    def test_comment_header_2(self) -> None:
        input = """

            // This is a comment

            OPENQASM 2.0;
            qreg q[1];
            u1(0.1) q[0];
        """
        circuit = OPENQASM2Language().decode(input)
        assert circuit.get_unitary() == U1Gate().get_unitary([0.1])

    def test_comment_footer_1(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            u1(0.1) q[0];
            // This is a comment
        """
        circuit = OPENQASM2Language().decode(input)
        assert circuit.get_unitary() == U1Gate().get_unitary([0.1])

    def test_comment_footer_2(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            u1(0.1) q[0];

            // This is a comment

        """
        circuit = OPENQASM2Language().decode(input)
        assert circuit.get_unitary() == U1Gate().get_unitary([0.1])

    def test_comment_gatedecl(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            gate gate_x q0 {
                // This is a commment
                u1(0.1) q0;
                // This is a comment too
            }
            gate_x q[0];
        """
        circuit = OPENQASM2Language().decode(input)
        assert circuit.get_unitary() == U1Gate().get_unitary([0.1])


class TestGateDecl:
    def test_simple_gate_decl(self) -> None:
        input = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            gate gate_x (p0) q0 {
                u2(p0, 3.5*p0) q0;
            }
            gate_x(1.2) q[0];
        """

        pytest.importorskip('qiskit')
        from qiskit import QuantumCircuit
        circuit = OPENQASM2Language().decode(input)
        assert circuit.num_qudits == 1
        assert circuit.num_operations == 1
        op = circuit[0, 0]
        assert isinstance(op.gate, CircuitGate)
        assert op.location == (0,)
        assert op.params == [1.2, 4.2]
        assert op.gate._circuit[0, 0].gate == U2Gate()
        assert op.gate._circuit[0, 0].location == (0,)

        qc = QuantumCircuit.from_qasm_str(input)
        bqskit_utry = circuit.get_unitary()
        qiskit_utry = qiskit_to_bqskit(qc).get_unitary()
        assert qiskit_utry.get_distance_from(bqskit_utry) < 1e-7

    def test_gate_decl_UCX(self) -> None:
        input = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            gate gate_x (p0) q0, q1 {
                U(p0, 2, 3) q0;
                CX q0, q1;
            }
            gate_x(1.2) q[0], q[1];
        """

        circuit = OPENQASM2Language().decode(input)
        assert circuit.num_operations == 1
        op = circuit[0, 0]
        assert isinstance(op.gate, CircuitGate)
        assert op.location == (0, 1)
        U1 = U3Gate().get_unitary([1.2, 2, 3]).otimes(UnitaryMatrix.identity(2))
        U2 = CNOTGate().get_unitary()
        assert circuit.get_unitary().get_distance_from(U2 @ U1) < 1e-7

    def test_empty_gate_decl_1(self) -> None:
        input = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            gate gate_x () q0 {
            }
            gate_x q[0];
        """

        pytest.importorskip('qiskit')
        from qiskit import QuantumCircuit
        circuit = OPENQASM2Language().decode(input)
        assert circuit.num_qudits == 1
        op = circuit[0, 0]
        assert isinstance(op.gate, CircuitGate)
        assert op.location == (0,)
        assert op.params == []

        qc = QuantumCircuit.from_qasm_str(input)
        bqskit_utry = circuit.get_unitary()
        qiskit_utry = qiskit_to_bqskit(qc).get_unitary()
        assert qiskit_utry.get_distance_from(bqskit_utry) < 1e-7

    def test_empty_gate_decl_2(self) -> None:
        input = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            gate gate_x q0 {}
            gate_x q[0];
        """

        pytest.importorskip('qiskit')
        from qiskit import QuantumCircuit

        circuit = OPENQASM2Language().decode(input)
        assert circuit.num_qudits == 1
        op = circuit[0, 0]
        assert isinstance(op.gate, CircuitGate)
        assert op.location == (0,)
        assert op.params == []

        qc = QuantumCircuit.from_qasm_str(input)
        bqskit_utry = circuit.get_unitary()
        qiskit_utry = qiskit_to_bqskit(qc).get_unitary()
        assert qiskit_utry.get_distance_from(bqskit_utry) < 1e-7

    def test_gate_decl_qubit_mixup(self) -> None:
        input = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
            gate gate_x q0, q1, q2 {
                cx q2, q1;
                cx q0, q2;
                cx q1, q0;
                cx q2, q0;
            }
            gate_x q[1], q[2], q[0];
        """

        pytest.importorskip('qiskit')
        from qiskit import QuantumCircuit

        circuit = OPENQASM2Language().decode(input)
        assert circuit.num_qudits == 3
        op = circuit[0, 0]
        assert isinstance(op.gate, CircuitGate)
        assert op.location == (1, 2, 0)
        assert op.params == []

        subcircuit = op.gate._circuit
        assert subcircuit.num_qudits == 3
        assert subcircuit.num_cycles == 4
        assert subcircuit[0, 1].gate == CNOTGate()
        assert subcircuit[0, 1].location == (2, 1)
        assert subcircuit[1, 0].gate == CNOTGate()
        assert subcircuit[1, 0].location == (0, 2)
        assert subcircuit[2, 1].gate == CNOTGate()
        assert subcircuit[2, 1].location == (1, 0)
        assert subcircuit[3, 2].gate == CNOTGate()
        assert subcircuit[3, 2].location == (2, 0)

        qc = QuantumCircuit.from_qasm_str(input)
        bqskit_utry = circuit.get_unitary()
        qiskit_utry = qiskit_to_bqskit(qc).get_unitary()
        assert qiskit_utry.get_distance_from(bqskit_utry) < 1e-7

    def test_nested_gate_decl(self) -> None:
        input = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            u1(0.1) q[0];
            gate gate_x (p0) q0 {
                u2(p0, 3.5*p0) q0;
            }
            gate gate_y (p0) q0, q1 {
                gate_x(p0) q0;
                u1(0.1) q0;
                gate_x(p0*2) q1;
            }
            gate_y(1.2) q[0], q[1];
        """

        circuit = OPENQASM2Language().decode(input)
        assert circuit.num_qudits == 2
        op = circuit[0, 0]
        assert op.gate == U1Gate()
        assert op.location == (0,)
        assert op.params == [0.1]

        op = circuit[1, 1]
        assert isinstance(op.gate, CircuitGate)
        assert op.location == (0, 1)

        subcircuit = op.gate._circuit
        assert isinstance(subcircuit[0, 0].gate, CircuitGate)
        assert subcircuit[0, 0].gate._circuit[0, 0].gate == U2Gate()
        assert subcircuit[0, 0].gate._circuit[0, 0].params == [1.2, 4.2]
        assert isinstance(subcircuit[0, 1].gate, CircuitGate)
        assert subcircuit[0, 1].gate._circuit[0, 0].gate == U2Gate()
        assert subcircuit[0, 1].gate._circuit[0, 0].params == [2.4, 8.4]
        assert subcircuit[1, 0].gate == U1Gate()
        assert subcircuit[1, 0].params == [0.1]

        # Unable to verify this one with qiskit
        # https://github.com/Qiskit/qiskit-terra/issues/8558
        # qc = QuantumCircuit.from_qasm_str(input)
        # bqskit_utry = circuit.get_unitary()
        # qiskit_utry = qiskit_to_bqskit(qc).get_unitary()
        # assert qiskit_utry.get_distance_from(bqskit_utry) < 1e-7


class TestIncludeStatements:
    def test_include_no_exists(self) -> None:
        input = """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
        """
        circuit = OPENQASM2Language().decode(input)
        assert circuit.num_qudits == 2
        assert circuit.num_operations == 0

    def test_include_simple(self) -> None:
        idata = """
            gate test(p) q { u1(p) q; }
        """
        input = """
            OPENQASM 2.0;
            include "test.inc";
            qreg q[1];
            test(0.1) q[0];
        """
        with patch('builtins.open', mock_open(read_data=idata)) as mock_file:
            with patch('os.path.isfile', lambda x: True):
                circuit = OPENQASM2Language().decode(input)
            mock_file.assert_called_with('test.inc')
        assert circuit.num_qudits == 1
        assert circuit.num_operations == 1
        gate_unitary = U1Gate().get_unitary([0.1])
        assert circuit.get_unitary().get_distance_from(gate_unitary) < 1e-7


class TestReset:
    def test_reset_single_qubit(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            reset q[0];
        """
        circuit = OPENQASM2Language().decode(input)
        expected = Reset()
        assert circuit[0, 0].gate == expected

    def test_reset_register(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[2];
            reset q;
        """
        circuit = OPENQASM2Language().decode(input)
        expected = Reset()
        assert circuit[0, 0].gate == expected
        assert circuit[0, 1].gate == expected

    def test_mid_reset(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            u1(0.1) q[0];
            reset q[0];
            u1(0.1) q[0];
        """
        circuit = OPENQASM2Language().decode(input)
        reset = Reset()
        assert circuit[0, 0].gate == U1Gate()
        assert circuit[1, 0].gate == reset
        assert circuit[2, 0].gate == U1Gate()


class TestMeasure:
    def test_measure_single_bit(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
        """
        circuit = OPENQASM2Language().decode(input)
        expected = MeasurementPlaceholder([('c', 1)], {0: ('c', 0)})
        assert circuit[0, 0].gate == expected

    def test_mid_measure_single_bit(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            creg c[1];
            u1(0.1) q[0];
            measure q[0] -> c[0];
            u1(0.1) q[0];
        """
        circuit = OPENQASM2Language().decode(input)
        measurements = {0: ('c', 0)}
        measure = MeasurementPlaceholder([('c', 1)], measurements)
        assert circuit[0, 0].gate == U1Gate()
        assert circuit[1, 0].gate == measure
        assert circuit[2, 0].gate == U1Gate()

    def test_mid_measure_register_1(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            creg c[1];
            u1(0.1) q[0];
            measure q -> c;
            u1(0.1) q[0];
        """
        circuit = OPENQASM2Language().decode(input)
        measurements = {0: ('c', 0)}
        measure = MeasurementPlaceholder([('c', 1)], measurements)
        assert circuit[0, 0].gate == U1Gate()
        assert circuit[1, 0].gate == measure
        assert circuit[2, 0].gate == U1Gate()

    def test_mid_measure_register_2(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[2];
            creg c[2];
            u1(0.1) q[0];
            measure q -> c;
            u1(0.1) q[0];
        """
        circuit = OPENQASM2Language().decode(input)
        measurements = {0: ('c', 0), 1: ('c', 1)}
        measure = MeasurementPlaceholder([('c', 2)], measurements)
        assert circuit[0, 0].gate == U1Gate()
        assert circuit[1, 0].gate == measure
        assert circuit[2, 0].gate == U1Gate()

    def test_measure_register_1(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[1];
            creg c[1];
            measure q -> c;
        """
        circuit = OPENQASM2Language().decode(input)
        expected = MeasurementPlaceholder([('c', 1)], {0: ('c', 0)})
        assert circuit[0, 0].gate == expected

    def test_measure_register_2(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[2];
            creg c[2];
            measure q -> c;
        """
        circuit = OPENQASM2Language().decode(input)
        measurements = {0: ('c', 0), 1: ('c', 1)}
        expected = MeasurementPlaceholder([('c', 2)], measurements)
        assert circuit[0, 0].gate == expected

    def test_measure_register_invalid_size(self) -> None:
        input = """
            OPENQASM 2.0;
            qreg q[2];
            creg c[1];
            measure q -> c;
        """
        with pytest.raises(LangException):
            circuit = OPENQASM2Language().decode(input)  # noqa


def test_U_gate() -> None:
    input = """
        OPENQASM 2.0;
        qreg q[1];
        U(1,2.1,3) q[0];
    """
    circuit = OPENQASM2Language().decode(input)
    assert circuit.num_operations == 1
    assert circuit.get_unitary() == U3Gate().get_unitary([1, 2.1, 3])


def test_CX_gate() -> None:
    input = """
        OPENQASM 2.0;
        qreg q[2];
        CX q[0],q[1];
    """
    circuit = OPENQASM2Language().decode(input)
    assert circuit.num_operations == 1
    assert circuit[0, 0].gate == CNOTGate()


def test_barrier_full_register() -> None:
    input = """
        OPENQASM 2.0;
        qreg q[2];
        CX q[0],q[1];
        barrier q;
        CX q[0],q[1];
    """
    circuit = OPENQASM2Language().decode(input)
    assert circuit.num_operations == 3
    assert circuit[0, 0].gate == CNOTGate()
    assert circuit[1, 0].gate == BarrierPlaceholder(2)
    assert circuit[2, 0].gate == CNOTGate()


def test_barrier_indiviual_qubits() -> None:
    input = """
        OPENQASM 2.0;
        qreg q[2];
        CX q[0],q[1];
        barrier q[0], q[1];
        CX q[0],q[1];
    """
    circuit = OPENQASM2Language().decode(input)
    assert circuit.num_operations == 3
    assert circuit[0, 0].gate == CNOTGate()
    assert circuit[1, 0].gate == BarrierPlaceholder(2)
    assert circuit[2, 0].gate == CNOTGate()


def test_barrier_mixed() -> None:
    input = """
        OPENQASM 2.0;
        qreg q[2];
        qreg r[2];
        CX q[0],q[1];
        barrier q, r[0];
        CX q[0],q[1];
    """
    circuit = OPENQASM2Language().decode(input)
    assert circuit.num_operations == 3
    assert circuit[0, 0].gate == CNOTGate()
    assert circuit[1, 0].gate == BarrierPlaceholder(3)
    assert circuit[2, 0].gate == CNOTGate()


def test_barrier_mixed_three() -> None:
    input = """
        OPENQASM 2.0;
        qreg q[2];
        qreg r[2];
        CX q[0],q[1];
        barrier q, r[0], r[1];
        CX q[0],q[1];
    """
    circuit = OPENQASM2Language().decode(input)
    assert circuit.num_operations == 3
    assert circuit[0, 0].gate == CNOTGate()
    assert circuit[1, 0].gate == BarrierPlaceholder(4)
    assert circuit[2, 0].gate == CNOTGate()


def test_U1Q_gate() -> None:
    input = """
        OPENQASM 2.0;
        qreg q[1];
        U1q(3.141592653589793, -0.8510194827063557) q[0];
        u1q(3.141592653589793, -0.8510194827063557) q[0];
    """
    circuit = OPENQASM2Language().decode(input)
    assert circuit.num_operations == 2
    assert circuit[0, 0].gate == U1qGate()
    assert circuit[1, 0].gate == U1qGate()


def test_ECR_gate() -> None:
    input = """
        OPENQASM 2.0;
        qreg q[2];
        ecr q[0],q[1];
    """
    circuit = OPENQASM2Language().decode(input)
    assert circuit.num_operations == 1
    assert circuit[0, 0].gate == ECRGate()

@pytest.mark.parametrize(
    'angles',
    [
        [-0.1, 0.2, -0.3],
        [1e-10, 1e-8, 2.2e-10],
        [-1e+4, 1e+5, -1e+6],
        [3.141592653589793, -2.718281828459045, 1.4142135623730951],
        [1, 2, 3],
        [1.0, 2.0, 3.0]
    ],
)
def test_decimal_angle(angles: list[float]) -> None:
    input_qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u3({angles[0]}, {angles[1]}, {angles[2]}) q[0];
    """
    circuit = OPENQASM2Language().decode(input_qasm)
    assert circuit.num_operations == 1
    u3_params = circuit[0, 0].params
    assert_almost_equal(u3_params, angles, decimal=8)
