"""
This test module verifies all circuit qudit methods.

Circuit qudit methods:
    def append_qudit(self, radix: int = 2) -> None
    def extend_qudits(self, radixes: Sequence[int]) -> None
    def insert_qudit(self, qudit_index: int, radix: int = 2) -> None
    def pop_qudit(self, qudit_index: int) -> None
    def is_qudit_in_range(self, qudit_index: int) -> bool
    def is_qudit_idle(self, qudit_index: int) -> bool
"""
from __future__ import annotations

from typing import Any
from typing import Sequence

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate


class TestAppendQubit:
    """This tests circuit.append_qudit."""

    def test_append_qudit_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_append_qudit_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append_qudit(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_append_qudit_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_append_qudit_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append_qudit(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_append_qudit_value_invalid1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(-1)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_append_qudit_value_invalid2(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(0)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_append_qudit_value_invalid3(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(1)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_append_qudit_default(self) -> None:
        circuit = Circuit(1)
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 2
        assert len(circuit.get_radixes()) == 1
        circuit.append_qudit()
        assert circuit.get_size() == 2
        assert circuit.get_dim() == 4
        assert len(circuit.get_radixes()) == 2
        circuit.append_qudit()
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 8
        assert len(circuit.get_radixes()) == 3

    def test_append_qudit_qutrit(self) -> None:
        circuit = Circuit(1, [3])
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 3
        assert len(circuit.get_radixes()) == 1
        circuit.append_qudit(3)
        assert circuit.get_size() == 2
        assert circuit.get_dim() == 9
        assert len(circuit.get_radixes()) == 2
        circuit.append_qudit(3)
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 27
        assert len(circuit.get_radixes()) == 3

    def test_append_qudit_hybrid(self) -> None:
        circuit = Circuit(1)
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 2
        assert len(circuit.get_radixes()) == 1
        circuit.append_qudit(4)
        assert circuit.get_size() == 2
        assert circuit.get_dim() == 8
        assert len(circuit.get_radixes()) == 2
        assert circuit.get_radixes()[0] == 2
        assert circuit.get_radixes()[1] == 4
        circuit.append_qudit(3)
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 24
        assert len(circuit.get_radixes()) == 3
        assert circuit.get_radixes()[0] == 2
        assert circuit.get_radixes()[1] == 4
        assert circuit.get_radixes()[2] == 3

    def test_append_qudit_append_gate(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_qudit()
        circuit.append_gate(CNOTGate(), [3, 4])
        assert circuit.get_size() == 5
        assert circuit[3, 4].gate == CNOTGate()


class TestExtendQubits:
    """This tests circuit.extend_qudits."""

    def test_extend_qudits_type_valid_1(
            self, a_seq_int: Sequence[int],
    ) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits(a_seq_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_extend_qudits_type_valid_2(
            self, a_seq_int: Sequence[int],
    ) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.extend_qudits(a_seq_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_extend_qudits_type_invalid_1(self, not_a_seq_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits(not_a_seq_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_qudits_type_invalid_2(self, not_a_seq_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.extend_qudits(not_a_seq_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_qudits_value_invalid1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([-1])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_qudits_value_invalid2(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([0])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_qudits_value_invalid3(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([1])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_qudits_value_invalid4(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([2, 2, -1])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_qudits_value_invalid5(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([2, 0, 2])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_qudits_value_invalid6(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([2, 2, 3, 1])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_extend_qudits_qubits(self) -> None:
        circuit = Circuit(1, [2])
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 2
        assert len(circuit.get_radixes()) == 1
        circuit.extend_qudits([2, 2])
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 8
        assert len(circuit.get_radixes()) == 3
        circuit.extend_qudits([2, 2])
        assert circuit.get_size() == 5
        assert circuit.get_dim() == 32
        assert len(circuit.get_radixes()) == 5

    def test_extend_qudits_qutrits(self) -> None:
        circuit = Circuit(1, [3])
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 3
        assert len(circuit.get_radixes()) == 1
        circuit.extend_qudits([3, 3])
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 27
        assert len(circuit.get_radixes()) == 3
        circuit.extend_qudits([3, 3])
        assert circuit.get_size() == 5
        assert circuit.get_dim() == 243
        assert len(circuit.get_radixes()) == 5

    def test_extend_qudits_hybrid(self) -> None:
        circuit = Circuit(1)
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 2
        assert len(circuit.get_radixes()) == 1
        circuit.extend_qudits([3, 4])
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 24
        assert len(circuit.get_radixes()) == 3
        assert circuit.get_radixes()[0] == 2
        assert circuit.get_radixes()[1] == 3
        assert circuit.get_radixes()[2] == 4
        circuit.extend_qudits([3, 2])
        assert circuit.get_size() == 5
        assert circuit.get_dim() == 144
        assert len(circuit.get_radixes()) == 5
        assert circuit.get_radixes()[0] == 2
        assert circuit.get_radixes()[1] == 3
        assert circuit.get_radixes()[2] == 4
        assert circuit.get_radixes()[3] == 3
        assert circuit.get_radixes()[4] == 2

    def test_extend_qudits_append_gate(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.extend_qudits([2, 2])
        circuit.append_gate(CNOTGate(), [3, 4])
        circuit.append_gate(CNOTGate(), [4, 5])
        assert circuit.get_size() == 6
        assert circuit[3, 4].gate == CNOTGate()
        assert circuit[4, 5].gate == CNOTGate()


class TestInsertQubit:
    """This tests circuit.insert_qudit."""

    def test_insert_qudit_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(an_int, an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_insert_qudit_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.insert_qudit(an_int, an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_insert_qudit_type_invalid_1(
            self, not_an_int: Any, an_int: int,
    ) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(not_an_int, an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_type_invalid_2(
            self, not_an_int: Any, an_int: int,
    ) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.insert_qudit(not_an_int, an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_type_invalid_3(
            self, not_an_int: Any, an_int: int,
    ) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_type_invalid_4(
            self, not_an_int: Any, an_int: int,
    ) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.insert_qudit(an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_type_invalid_5(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(not_an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_type_invalid_6(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.insert_qudit(not_an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_value_invalid1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(0, -1)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_value_invalid2(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(0, 0)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_value_invalid3(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(0, 1)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_insert_qudit_default(self) -> None:
        circuit = Circuit(1)
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 2
        assert len(circuit.get_radixes()) == 1
        circuit.insert_qudit(0)
        assert circuit.get_size() == 2
        assert circuit.get_dim() == 4
        assert len(circuit.get_radixes()) == 2
        circuit.insert_qudit(0)
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 8
        assert len(circuit.get_radixes()) == 3

    def test_insert_qudit_qutrit(self) -> None:
        circuit = Circuit(1, [3])
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 3
        assert len(circuit.get_radixes()) == 1
        circuit.insert_qudit(0, 3)
        assert circuit.get_size() == 2
        assert circuit.get_dim() == 9
        assert len(circuit.get_radixes()) == 2
        circuit.insert_qudit(0, 3)
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 27
        assert len(circuit.get_radixes()) == 3

    def test_insert_qudit_hybrid(self) -> None:
        circuit = Circuit(1)
        assert circuit.get_size() == 1
        assert circuit.get_dim() == 2
        assert len(circuit.get_radixes()) == 1
        circuit.insert_qudit(0, 4)
        assert circuit.get_size() == 2
        assert circuit.get_dim() == 8
        assert len(circuit.get_radixes()) == 2
        assert circuit.get_radixes()[0] == 4
        assert circuit.get_radixes()[1] == 2
        circuit.insert_qudit(-1, 3)
        assert circuit.get_size() == 3
        assert circuit.get_dim() == 24
        assert len(circuit.get_radixes()) == 3
        assert circuit.get_radixes()[0] == 4
        assert circuit.get_radixes()[1] == 3
        assert circuit.get_radixes()[2] == 2

    def test_insert_qudit_append_gate(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(0)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.get_size() == 5
        assert circuit[3, 0].gate == CNOTGate()
        assert circuit[3, 0].location == (0, 3)
        assert circuit[0, 1].gate == CNOTGate()
        assert circuit[0, 1].location == (1, 2)
        assert circuit[1, 2].gate == CNOTGate()
        assert circuit[1, 2].location == (2, 3)
        assert circuit[2, 3].gate == CNOTGate()
        assert circuit[2, 3].location == (3, 4)
