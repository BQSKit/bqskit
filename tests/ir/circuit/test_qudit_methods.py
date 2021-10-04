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

import numpy as np
import pytest

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.ir.gates import HGate


class TestAppendQudit:
    """This tests `circuit.append_qudit`."""

    def test_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append_qudit(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.append_qudit(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(-1)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid2(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(0)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid3(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.append_qudit(1)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_default(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_qudits == 1
        assert circuit.dim == 2
        assert len(circuit.radixes) == 1
        circuit.append_qudit()
        assert circuit.num_qudits == 2
        assert circuit.dim == 4
        assert len(circuit.radixes) == 2
        circuit.append_qudit()
        assert circuit.num_qudits == 3
        assert circuit.dim == 8
        assert len(circuit.radixes) == 3

    def test_qutrit(self) -> None:
        circuit = Circuit(1, [3])
        assert circuit.num_qudits == 1
        assert circuit.dim == 3
        assert len(circuit.radixes) == 1
        circuit.append_qudit(3)
        assert circuit.num_qudits == 2
        assert circuit.dim == 9
        assert len(circuit.radixes) == 2
        circuit.append_qudit(3)
        assert circuit.num_qudits == 3
        assert circuit.dim == 27
        assert len(circuit.radixes) == 3

    def test_hybrid(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_qudits == 1
        assert circuit.dim == 2
        assert len(circuit.radixes) == 1
        circuit.append_qudit(4)
        assert circuit.num_qudits == 2
        assert circuit.dim == 8
        assert len(circuit.radixes) == 2
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 4
        circuit.append_qudit(3)
        assert circuit.num_qudits == 3
        assert circuit.dim == 24
        assert len(circuit.radixes) == 3
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 4
        assert circuit.radixes[2] == 3

    def test_append_gate(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_qudit()
        circuit.append_gate(CNOTGate(), [3, 4])
        assert circuit.num_qudits == 5
        assert circuit[3, 4].gate == CNOTGate()


class TestExtendQudits:
    """This tests `circuit.extend_qudits`."""

    def test_type_valid_1(self, a_seq_int: Sequence[int]) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits(a_seq_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_2(self, a_seq_int: Sequence[int]) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.extend_qudits(a_seq_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_invalid_1(self, not_a_seq_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits(not_a_seq_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_2(self, not_a_seq_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.extend_qudits(not_a_seq_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([-1])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid2(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([0])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid3(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([1])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid4(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([2, 2, -1])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid5(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([2, 0, 2])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid6(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.extend_qudits([2, 2, 3, 1])
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_qubits(self) -> None:
        circuit = Circuit(1, [2])
        assert circuit.num_qudits == 1
        assert circuit.dim == 2
        assert len(circuit.radixes) == 1
        circuit.extend_qudits([2, 2])
        assert circuit.num_qudits == 3
        assert circuit.dim == 8
        assert len(circuit.radixes) == 3
        circuit.extend_qudits([2, 2])
        assert circuit.num_qudits == 5
        assert circuit.dim == 32
        assert len(circuit.radixes) == 5

    def test_qutrits(self) -> None:
        circuit = Circuit(1, [3])
        assert circuit.num_qudits == 1
        assert circuit.dim == 3
        assert len(circuit.radixes) == 1
        circuit.extend_qudits([3, 3])
        assert circuit.num_qudits == 3
        assert circuit.dim == 27
        assert len(circuit.radixes) == 3
        circuit.extend_qudits([3, 3])
        assert circuit.num_qudits == 5
        assert circuit.dim == 243
        assert len(circuit.radixes) == 5

    def test_hybrid(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_qudits == 1
        assert circuit.dim == 2
        assert len(circuit.radixes) == 1
        circuit.extend_qudits([3, 4])
        assert circuit.num_qudits == 3
        assert circuit.dim == 24
        assert len(circuit.radixes) == 3
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 3
        assert circuit.radixes[2] == 4
        circuit.extend_qudits([3, 2])
        assert circuit.num_qudits == 5
        assert circuit.dim == 144
        assert len(circuit.radixes) == 5
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 3
        assert circuit.radixes[2] == 4
        assert circuit.radixes[3] == 3
        assert circuit.radixes[4] == 2

    def test_append_gate(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.extend_qudits([2, 2])
        circuit.append_gate(CNOTGate(), [3, 4])
        circuit.append_gate(CNOTGate(), [4, 5])
        assert circuit.num_qudits == 6
        assert circuit[3, 4].gate == CNOTGate()
        assert circuit[4, 5].gate == CNOTGate()


class TestInsertQudit:
    """This tests `circuit.insert_qudit`."""

    def test_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(an_int, an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.insert_qudit(an_int, an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_invalid_1(self, not_an_int: Any, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(not_an_int, an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_2(self, not_an_int: Any, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.insert_qudit(not_an_int, an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_3(self, not_an_int: Any, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_4(self, not_an_int: Any, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.insert_qudit(an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_5(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(not_an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_6(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.insert_qudit(not_an_int, not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid_1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(0, -1)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid_2(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(0, 0)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_value_invalid_3(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.insert_qudit(0, 1)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_default(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_qudits == 1
        assert circuit.dim == 2
        assert len(circuit.radixes) == 1
        circuit.insert_qudit(0)
        assert circuit.num_qudits == 2
        assert circuit.dim == 4
        assert len(circuit.radixes) == 2
        circuit.insert_qudit(0)
        assert circuit.num_qudits == 3
        assert circuit.dim == 8
        assert len(circuit.radixes) == 3

    def test_qutrit(self) -> None:
        circuit = Circuit(1, [3])
        assert circuit.num_qudits == 1
        assert circuit.dim == 3
        assert len(circuit.radixes) == 1
        circuit.insert_qudit(0, 3)
        assert circuit.num_qudits == 2
        assert circuit.dim == 9
        assert len(circuit.radixes) == 2
        circuit.insert_qudit(0, 3)
        assert circuit.num_qudits == 3
        assert circuit.dim == 27
        assert len(circuit.radixes) == 3

    def test_hybrid(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_qudits == 1
        assert circuit.dim == 2
        assert len(circuit.radixes) == 1
        circuit.insert_qudit(0, 4)
        assert circuit.num_qudits == 2
        assert circuit.dim == 8
        assert len(circuit.radixes) == 2
        assert circuit.radixes[0] == 4
        assert circuit.radixes[1] == 2
        circuit.insert_qudit(-1, 3)
        assert circuit.num_qudits == 3
        assert circuit.dim == 24
        assert len(circuit.radixes) == 3
        assert circuit.radixes[0] == 4
        assert circuit.radixes[1] == 3
        assert circuit.radixes[2] == 2

    def test_append_gate_1(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(0)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[3, 0].gate == CNOTGate()
        assert circuit[3, 0].location == (0, 3)
        assert circuit[0, 1].gate == CNOTGate()
        assert circuit[0, 1].location == (1, 2)
        assert circuit[1, 2].gate == CNOTGate()
        assert circuit[1, 2].location == (2, 3)
        assert circuit[2, 3].gate == CNOTGate()
        assert circuit[2, 3].location == (3, 4)

    def test_append_gate_2(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(1)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[3, 0].gate == CNOTGate()
        assert circuit[3, 0].location == (0, 3)
        assert circuit[0, 0].gate == CNOTGate()
        assert circuit[0, 0].location == (0, 2)
        assert circuit[1, 2].gate == CNOTGate()
        assert circuit[1, 2].location == (2, 3)
        assert circuit[2, 3].gate == CNOTGate()
        assert circuit[2, 3].location == (3, 4)

    def test_append_gate_3(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(2)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[3, 0].gate == CNOTGate()
        assert circuit[3, 0].location == (0, 3)
        assert circuit[0, 0].gate == CNOTGate()
        assert circuit[0, 0].location == (0, 1)
        assert circuit[1, 1].gate == CNOTGate()
        assert circuit[1, 1].location == (1, 3)
        assert circuit[2, 3].gate == CNOTGate()
        assert circuit[2, 3].location == (3, 4)

    def test_append_gate_4(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(3)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[1, 0].gate == CNOTGate()
        assert circuit[1, 0].location == (0, 3)
        assert circuit[0, 0].gate == CNOTGate()
        assert circuit[0, 0].location == (0, 1)
        assert circuit[1, 1].gate == CNOTGate()
        assert circuit[1, 1].location == (1, 2)
        assert circuit[2, 2].gate == CNOTGate()
        assert circuit[2, 2].location == (2, 4)

    def test_append_gate_5(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(4)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[3, 0].gate == CNOTGate()
        assert circuit[3, 0].location == (0, 3)
        assert circuit[0, 1].gate == CNOTGate()
        assert circuit[0, 1].location == (0, 1)
        assert circuit[1, 2].gate == CNOTGate()
        assert circuit[1, 2].location == (1, 2)
        assert circuit[2, 3].gate == CNOTGate()
        assert circuit[2, 3].location == (2, 3)
        assert circuit[3, 3].gate == CNOTGate()
        assert circuit[3, 3].location == (0, 3)
        assert circuit[0, 0].gate == CNOTGate()
        assert circuit[0, 0].location == (0, 1)
        assert circuit[1, 1].gate == CNOTGate()
        assert circuit[1, 1].location == (1, 2)
        assert circuit[2, 2].gate == CNOTGate()
        assert circuit[2, 2].location == (2, 3)

    def test_append_gate_6(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(-3)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[3, 0].gate == CNOTGate()
        assert circuit[3, 0].location == (0, 3)
        assert circuit[0, 0].gate == CNOTGate()
        assert circuit[0, 0].location == (0, 2)
        assert circuit[1, 2].gate == CNOTGate()
        assert circuit[1, 2].location == (2, 3)
        assert circuit[2, 3].gate == CNOTGate()
        assert circuit[2, 3].location == (3, 4)

    def test_append_gate_7(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(-6)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[3, 0].gate == CNOTGate()
        assert circuit[3, 0].location == (0, 3)
        assert circuit[0, 1].gate == CNOTGate()
        assert circuit[0, 1].location == (1, 2)
        assert circuit[1, 2].gate == CNOTGate()
        assert circuit[1, 2].location == (2, 3)
        assert circuit[2, 3].gate == CNOTGate()
        assert circuit[2, 3].location == (3, 4)

    def test_append_gate_8(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.insert_qudit(25)
        circuit.append_gate(CNOTGate(), [0, 3])
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[3, 0].gate == CNOTGate()
        assert circuit[3, 0].location == (0, 3)
        assert circuit[0, 1].gate == CNOTGate()
        assert circuit[0, 1].location == (0, 1)
        assert circuit[1, 2].gate == CNOTGate()
        assert circuit[1, 2].location == (1, 2)
        assert circuit[2, 3].gate == CNOTGate()
        assert circuit[2, 3].location == (2, 3)

    def test_multi_gate_1(self, gen_random_utry_np: Any) -> None:
        circuit = Circuit(4)
        three_qubit_gate = ConstantUnitaryGate(gen_random_utry_np(8))
        circuit.append_gate(three_qubit_gate, [1, 2, 3])
        circuit.append_gate(three_qubit_gate, [0, 2, 3])
        circuit.append_gate(three_qubit_gate, [0, 1, 3])
        circuit.append_gate(three_qubit_gate, [0, 1, 2])
        circuit.insert_qudit(0)
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes.count(2) == 5
        assert circuit[0, 2].gate == three_qubit_gate
        assert circuit[0, 2].location == (2, 3, 4)
        assert circuit[1, 1].gate == three_qubit_gate
        assert circuit[1, 1].location == (1, 3, 4)
        assert circuit[2, 1].gate == three_qubit_gate
        assert circuit[2, 1].location == (1, 2, 4)
        assert circuit[3, 1].gate == three_qubit_gate
        assert circuit[3, 1].location == (1, 2, 3)

    def test_multi_gate_2(self, gen_random_utry_np: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        three_qubit_gate = ConstantUnitaryGate(
            gen_random_utry_np(12), [2, 2, 3],
        )
        circuit.append_gate(three_qubit_gate, [0, 1, 3])
        circuit.append_gate(three_qubit_gate, [0, 1, 2])
        circuit.insert_qudit(2)
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 2
        assert circuit.radixes[2] == 2
        assert circuit.radixes[3] == 3
        assert circuit.radixes[4] == 3
        assert circuit[0, 1].gate == three_qubit_gate
        assert circuit[0, 1].location == (0, 1, 4)
        assert circuit[1, 1].gate == three_qubit_gate
        assert circuit[1, 1].location == (0, 1, 3)

    def test_multi_gate_3(self, gen_random_utry_np: Any) -> None:
        circuit = Circuit(4)
        three_qubit_gate = ConstantUnitaryGate(
            gen_random_utry_np(12), [2, 2, 3],
        )
        circuit.insert_qudit(2, 3)
        assert circuit.num_qudits == 5
        assert len(circuit.radixes) == 5
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 2
        assert circuit.radixes[2] == 3
        assert circuit.radixes[3] == 2
        assert circuit.radixes[4] == 2
        circuit.append_gate(three_qubit_gate, [0, 1, 2])
        assert circuit[0, 0].gate == three_qubit_gate
        assert circuit[0, 0].location == (0, 1, 2)


class TestPopQudit:
    """This tests `circuit.pop_qudit`."""

    def test_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.pop_qudit(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.pop_qudit(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.pop_qudit(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.pop_qudit(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_index_invalid1(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.pop_qudit(-5)
        except IndexError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_index_invalid2(self) -> None:
        circuit = Circuit(1)
        try:
            circuit.pop_qudit(5)
        except IndexError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_index_invalid3(self) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.pop_qudit(5)
        except IndexError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_index_invalid_empty(self) -> None:
        circuit = Circuit(1)
        assert circuit.num_qudits == 1
        try:
            circuit.pop_qudit(0)
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    @pytest.mark.parametrize('qudit_index', [-4, -1, 0, 3])
    def test_append_gate_1(self, qudit_index: int) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.pop_qudit(qudit_index)
        assert circuit.num_qudits == 3
        assert len(circuit.radixes) == 3
        assert circuit.radixes.count(2) == 3
        assert circuit.get_num_operations() == 2
        assert circuit[0, 0].gate == CNOTGate()
        assert circuit[0, 0].location == (0, 1)
        assert circuit[1, 1].gate == CNOTGate()
        assert circuit[1, 1].location == (1, 2)
        assert circuit[0, 1].gate == CNOTGate()
        assert circuit[0, 1].location == (0, 1)
        assert circuit[1, 2].gate == CNOTGate()
        assert circuit[1, 2].location == (1, 2)

    @pytest.mark.parametrize('qudit_index', [-3, 1])
    def test_append_gate_2(self, qudit_index: int) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.pop_qudit(qudit_index)
        assert circuit.num_qudits == 3
        assert len(circuit.radixes) == 3
        assert circuit.radixes.count(2) == 3
        assert circuit.get_num_operations() == 1
        assert circuit[0, 1].gate == CNOTGate()
        assert circuit[0, 1].location == (1, 2)
        assert circuit[0, 2].gate == CNOTGate()
        assert circuit[0, 2].location == (1, 2)

    @pytest.mark.parametrize('qudit_index', [-2, 2])
    def test_append_gate_3(self, qudit_index: int) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.pop_qudit(qudit_index)
        assert circuit.num_qudits == 3
        assert len(circuit.radixes) == 3
        assert circuit.radixes.count(2) == 3
        assert circuit.get_num_operations() == 1
        assert circuit[0, 0].gate == CNOTGate()
        assert circuit[0, 0].location == (0, 1)
        assert circuit[0, 1].gate == CNOTGate()
        assert circuit[0, 1].location == (0, 1)

    @pytest.mark.parametrize('qudit_index', [-4, -1, 0, 3])
    def test_append_gate_4(self, qudit_index: int) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(CNOTGate(), [2, 3])
        circuit.pop_qudit(qudit_index)
        assert circuit.num_qudits == 3
        assert len(circuit.radixes) == 3
        assert circuit.radixes.count(2) == 3
        assert circuit.get_num_operations() == 6
        assert circuit[0, 0].gate == CNOTGate()
        assert circuit[0, 0].location == (0, 1)
        assert circuit[1, 1].gate == CNOTGate()
        assert circuit[1, 1].location == (1, 2)
        assert circuit[2, 0].gate == CNOTGate()
        assert circuit[2, 0].location == (0, 1)
        assert circuit[3, 1].gate == CNOTGate()
        assert circuit[3, 1].location == (1, 2)
        assert circuit[4, 0].gate == CNOTGate()
        assert circuit[4, 0].location == (0, 1)
        assert circuit[5, 1].gate == CNOTGate()
        assert circuit[5, 1].location == (1, 2)

    @pytest.mark.parametrize('qudit_index', [-4, -3, -2, -1, 0, 1, 2, 3])
    def test_multi_gate_1(
            self, qudit_index: int, gen_random_utry_np: Any,
    ) -> None:
        circuit = Circuit(4)
        three_qubit_gate = ConstantUnitaryGate(gen_random_utry_np(8))
        circuit.append_gate(three_qubit_gate, [1, 2, 3])
        circuit.append_gate(three_qubit_gate, [0, 2, 3])
        circuit.append_gate(three_qubit_gate, [0, 1, 3])
        circuit.append_gate(three_qubit_gate, [0, 1, 2])
        circuit.pop_qudit(qudit_index)
        assert circuit.num_qudits == 3
        assert len(circuit.radixes) == 3
        assert circuit.radixes.count(2) == 3
        assert circuit.get_num_operations() == 1
        assert circuit.get_num_cycles() == 1
        assert circuit[0, 0].gate == three_qubit_gate
        assert circuit[0, 0].location == (0, 1, 2)
        assert circuit[0, 1].gate == three_qubit_gate
        assert circuit[0, 1].location == (0, 1, 2)
        assert circuit[0, 2].gate == three_qubit_gate
        assert circuit[0, 2].location == (0, 1, 2)

    @pytest.mark.parametrize('qudit_index', [-2, -1, 2, 3])
    def test_multi_gate_2(
            self, qudit_index: int, gen_random_utry_np: Any,
    ) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        three_qubit_gate = ConstantUnitaryGate(
            gen_random_utry_np(12), [2, 2, 3],
        )
        circuit.append_gate(three_qubit_gate, [0, 1, 3])
        circuit.append_gate(three_qubit_gate, [0, 1, 2])
        circuit.pop_qudit(qudit_index)
        assert circuit.num_qudits == 3
        assert len(circuit.radixes) == 3
        assert circuit.get_num_operations() == 1
        assert circuit.get_num_cycles() == 1
        assert circuit.radixes[0] == 2
        assert circuit.radixes[1] == 2
        assert circuit.radixes[2] == 3
        assert circuit[0, 0].gate == three_qubit_gate
        assert circuit[0, 0].location == (0, 1, 2)
        assert circuit[0, 1].gate == three_qubit_gate
        assert circuit[0, 1].location == (0, 1, 2)
        assert circuit[0, 2].gate == three_qubit_gate
        assert circuit[0, 2].location == (0, 1, 2)


class TestIsQuditInRange:
    """This tests `circuit.is_qudit_in_range`."""

    def test_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_qudit_in_range(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.is_qudit_in_range(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_qudit_in_range(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.is_qudit_in_range(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_return_type(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        assert isinstance(circuit.is_qudit_in_range(an_int), (bool, np.bool_))

    @pytest.mark.parametrize('test_value', [-5, -4, -3, -2, -1])
    def test_true_neg(self, test_value: int) -> None:
        circuit = Circuit(5)
        assert circuit.is_qudit_in_range(test_value)

    @pytest.mark.parametrize('test_value', [0, 1, 2, 3, 4])
    def test_true_pos(self, test_value: int) -> None:
        circuit = Circuit(5)
        assert circuit.is_qudit_in_range(test_value)

    @pytest.mark.parametrize('test_value', [-1000, -100, -8, -6])
    def test_false_neg(self, test_value: int) -> None:
        circuit = Circuit(5)
        assert not circuit.is_qudit_in_range(test_value)

    @pytest.mark.parametrize('test_value', [5, 6, 8, 100, 1000])
    def test_false_pos(self, test_value: int) -> None:
        circuit = Circuit(5)
        assert not circuit.is_qudit_in_range(test_value)


class TestIsQuditIdle:
    """This tests `circuit.is_qudit_idle`."""

    def test_type_valid_1(self, an_int: int) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_qudit_idle(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_valid_2(self, an_int: int) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.is_qudit_idle(an_int)
        except TypeError:
            assert False, 'Unexpected TypeError.'
        except BaseException:
            return

    def test_type_invalid_1(self, not_an_int: Any) -> None:
        circuit = Circuit(1)
        try:
            circuit.is_qudit_idle(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_type_invalid_2(self, not_an_int: Any) -> None:
        circuit = Circuit(4, [2, 2, 3, 3])
        try:
            circuit.is_qudit_idle(not_an_int)
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected Exception.'

    def test_return_type(
            self, r6_qudit_circuit: Circuit,
    ) -> None:
        for i in range(6):
            assert isinstance(r6_qudit_circuit.is_qudit_idle(i), bool)

    def test_single(self) -> None:
        circuit = Circuit(4, [2, 2, 2, 2])
        assert circuit.is_qudit_idle(0)
        assert circuit.is_qudit_idle(1)
        assert circuit.is_qudit_idle(2)
        assert circuit.is_qudit_idle(3)

        circuit.append_gate(HGate(), [0])
        assert not circuit.is_qudit_idle(0)
        assert circuit.is_qudit_idle(1)
        assert circuit.is_qudit_idle(2)
        assert circuit.is_qudit_idle(3)

        circuit.append_gate(HGate(), [1])
        assert not circuit.is_qudit_idle(0)
        assert not circuit.is_qudit_idle(1)
        assert circuit.is_qudit_idle(2)
        assert circuit.is_qudit_idle(3)

        circuit.append_gate(HGate(), [2])
        assert not circuit.is_qudit_idle(0)
        assert not circuit.is_qudit_idle(1)
        assert not circuit.is_qudit_idle(2)
        assert circuit.is_qudit_idle(3)

        circuit.append_gate(HGate(), [3])
        assert not circuit.is_qudit_idle(0)
        assert not circuit.is_qudit_idle(1)
        assert not circuit.is_qudit_idle(2)
        assert not circuit.is_qudit_idle(3)

        circuit.pop((0, 0))
        assert circuit.is_qudit_idle(0)
        assert not circuit.is_qudit_idle(1)
        assert not circuit.is_qudit_idle(2)
        assert not circuit.is_qudit_idle(3)

        circuit.pop((0, 1))
        assert circuit.is_qudit_idle(0)
        assert circuit.is_qudit_idle(1)
        assert not circuit.is_qudit_idle(2)
        assert not circuit.is_qudit_idle(3)

        circuit.pop((0, 2))
        assert circuit.is_qudit_idle(0)
        assert circuit.is_qudit_idle(1)
        assert circuit.is_qudit_idle(2)
        assert not circuit.is_qudit_idle(3)

        circuit.pop((0, 3))
        assert circuit.is_qudit_idle(0)
        assert circuit.is_qudit_idle(1)
        assert circuit.is_qudit_idle(2)
        assert circuit.is_qudit_idle(3)
