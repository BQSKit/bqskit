from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TestQuickPartitioner:
    def test_run_r6(self, r6_qudit_circuit: Circuit) -> None:
        utry = r6_qudit_circuit.get_unitary()
        QuickPartitioner(3).run(r6_qudit_circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in r6_qudit_circuit
        )
        assert r6_qudit_circuit.get_unitary() == utry
        for cycle_index in range(r6_qudit_circuit.num_cycles):
            assert not r6_qudit_circuit._is_cycle_idle(cycle_index)

    def test_corner_case_1(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                1, 4,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 0,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(4),
            ), [
                2, 0, 3, 5,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_2(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [0])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                3, 0,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 0,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(4),
            ), [
                4, 0, 1, 2,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                5, 0, 1,
            ],
        )

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_3(self) -> None:
        circuit = Circuit(6)
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [0])
        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 1,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [0])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                5, 1, 3,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                4, 1,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                1, 3,
            ],
        )
        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_4(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                4, 0,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                2, 0,
            ],
        )
        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 3,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                1, 0,
            ],
        )

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_5(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                4, 1,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [4])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(4),
            ), [
                4, 1, 2, 3,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                4, 1,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 1,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                2, 1,
            ],
        )

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_6(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                1, 5,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                4, 0, 2,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [0])
        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 1,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(4),
            ), [
                0, 2, 3, 4,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                1, 5,
            ],
        )

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_7(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                0, 1,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [4])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                0, 1, 3,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 0,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                0, 1,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                4, 0, 1,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                1, 0, 3,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                4, 0,
            ],
        )

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_8(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 0,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [4])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                1, 0, 2,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                5, 0, 1,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                4, 0,
            ],
        )

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [4])

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [4])

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_9(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [0])
        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(3),
            ), [
                2, 1, 4,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                5, 1,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [2])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [4])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(4),
            ), [
                5, 1, 2, 4,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                4, 1,
            ],
        )

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_10(self) -> None:
        circuit = Circuit(6)

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [0])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [1])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [4])
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                1, 2,
            ],
        )
        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [5])

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                4, 1,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(2),
            ), [
                3, 1,
            ],
        )

        circuit.append_gate(
            ConstantUnitaryGate(
                UnitaryMatrix.random(4),
            ), [
                1, 2, 3, 4,
            ],
        )

        circuit.append_gate(ConstantUnitaryGate(UnitaryMatrix.random(1)), [3])

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.get_unitary() == utry
        for cycle_index in range(circuit.num_cycles):
            assert not circuit._is_cycle_idle(cycle_index)

    def test_corner_case_11(self) -> None:
        circuit = Circuit(4)

        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (2, 3))

        circuit.append_gate(CNOTGate(), (0, 3))
        circuit.append_gate(CNOTGate(), (1, 2))

        utry = circuit.get_unitary()
        QuickPartitioner(3).run(circuit, {})

        assert all(
            isinstance(op.gate, CircuitGate)
            or len(op.location) > 3
            for op in circuit
        )
        assert circuit.num_operations == 3
        assert circuit.get_unitary() == utry
