"""
This test module verifies all conftest fixtures work as intended.

The BQSKit root conftest defines the following circuit fixtures that
need to be verified:
    gen_random_utry_np
    gen_invalid_utry_np
    gen_random_circuit

There are four sets of randomly-generated circuits that are used to test
`gen_random_circuit` by proxy:
    r3_qubit_circuit
    r3_qubit_constant_circuit
    r3_qutrit_circuit
    r6_qudit_circuit

The BQSKit root conftest also defines several type-categorized value
fixtures that need to be verified:
    a_str
    not_a_str
    an_int
    not_an_int
    a_float
    not_a_float
    a_complex
    not_a_complex
    a_bool
    not_a_bool
    a_seq_str
    not_a_seq_str
    a_seq_int
    not_a_seq_int
    a_seq_float
    not_a_seq_float
    a_seq_complex
    not_a_seq_complex
    a_seq_bool
    not_a_seq_bool
"""
from __future__ import annotations

from typing import Any

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.utils.typing import is_complex
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_numeric
from bqskit.utils.typing import is_sequence


class TestGenRandomUtryNp:
    """Ensure random utry generator behaves as expected."""

    def test_invalid_1(self, gen_random_utry_np: Any) -> None:
        try:
            utry = gen_random_utry_np('a')  # noqa
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    def test_invalid_2(self, gen_random_utry_np: Any) -> None:
        try:
            utry = gen_random_utry_np(['a', 3])  # noqa
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    def test_valid_single_dim(self, gen_random_utry_np: Any) -> None:
        utry = gen_random_utry_np(8)
        assert isinstance(utry, np.ndarray)
        assert utry.shape == (8, 8)
        assert UnitaryMatrix.is_unitary(utry)

    def test_valid_multi_dim(self, gen_random_utry_np: Any) -> None:
        utry = gen_random_utry_np([4, 8])
        assert isinstance(utry, np.ndarray)
        assert utry.shape == (8, 8) or utry.shape == (4, 4)
        assert UnitaryMatrix.is_unitary(utry)


class TestGenInvalidUtryNp:
    """Ensure invalid utry generator behaves as expected."""

    def test_invalid_1(self, gen_invalid_utry_np: Any) -> None:
        try:
            iutry = gen_invalid_utry_np('a')  # noqa
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    def test_invalid_2(self, gen_invalid_utry_np: Any) -> None:
        try:
            iutry = gen_invalid_utry_np(['a', 3])  # noqa
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    def test_valid_single_dim(self, gen_invalid_utry_np: Any) -> None:
        iutry = gen_invalid_utry_np(8)
        assert isinstance(iutry, np.ndarray)
        assert iutry.shape == (8, 8)
        assert not UnitaryMatrix.is_unitary(iutry)

    def test_valid_multi_dim(self, gen_invalid_utry_np: Any) -> None:
        iutry = gen_invalid_utry_np([4, 8])
        assert isinstance(iutry, np.ndarray)
        assert iutry.shape == (8, 8) or iutry.shape == (4, 4)
        assert not UnitaryMatrix.is_unitary(iutry)


class TestGenRandomCircuit:
    """Ensure random circuit generator behaves as expected."""

    def test_invalid_type_1(self, gen_random_circuit: Any) -> None:
        try:
            circuit = gen_random_circuit('a')  # noqa
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    def test_invalid_type_2(self, gen_random_circuit: Any) -> None:
        try:
            circuit = gen_random_circuit(3, 'a')  # noqa
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    def test_invalid_type_3(self, gen_random_circuit: Any) -> None:
        try:
            circuit = gen_random_circuit(3, [2, 2, 2], 'a')  # noqa
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    def test_invalid_type_4(self, gen_random_circuit: Any) -> None:
        try:
            circuit = gen_random_circuit(3, [2, 2, 2], 5, 'a')  # noqa
        except TypeError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    def test_invalid_value_1(self, gen_random_circuit: Any) -> None:
        try:
            circuit = gen_random_circuit(3, [2, 2])  # noqa
        except ValueError:
            return
        except BaseException:
            assert False, 'Unexpected error.'

    # r3_qubit_circuit
    def test_r3_size(self, r3_qubit_circuit: Circuit) -> None:
        assert r3_qubit_circuit.num_qudits == 3

    def test_r3_radix(self, r3_qubit_circuit: Circuit) -> None:
        assert r3_qubit_circuit.is_qubit_only()

    def test_r3_depth(self, r3_qubit_circuit: Circuit) -> None:
        assert r3_qubit_circuit.num_operations == 10

    # r3_qubit_constant_circuit
    def test_r3_con_size(self, r3_qubit_constant_circuit: Circuit) -> None:
        assert r3_qubit_constant_circuit.num_qudits == 3

    def test_r3_con_radix(self, r3_qubit_constant_circuit: Circuit) -> None:
        assert r3_qubit_constant_circuit.is_qubit_only()

    def test_r3_con_depth(self, r3_qubit_constant_circuit: Circuit) -> None:
        assert r3_qubit_constant_circuit.num_operations == 25

    def test_r3_con_constant(self, r3_qubit_constant_circuit: Circuit) -> None:
        assert r3_qubit_constant_circuit.is_constant()

    # r3_qutrit_circuit
    def test_r3_qutrit_size(self, r3_qutrit_circuit: Circuit) -> None:
        assert r3_qutrit_circuit.num_qudits == 3

    def test_r3_qutrit_radix(self, r3_qutrit_circuit: Circuit) -> None:
        assert r3_qutrit_circuit.is_qutrit_only()

    def test_r3_qutrit_depth(self, r3_qutrit_circuit: Circuit) -> None:
        assert r3_qutrit_circuit.num_operations == 10

    # r6_qudit_circuit
    def test_r6_size(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.num_qudits == 6

    def test_r6_radix(self, r6_qudit_circuit: Circuit) -> None:
        count = r6_qudit_circuit.radixes.count(2)
        count += r6_qudit_circuit.radixes.count(3)
        assert count == r6_qudit_circuit.num_qudits

    def test_r6_depth(self, r6_qudit_circuit: Circuit) -> None:
        assert r6_qudit_circuit.num_operations == 10


class TestTypedValues:
    """This tests the type-categorized value fixtures."""

    def test_a_str(self, a_str: Any) -> None:
        assert isinstance(a_str, str)

    def test_not_a_str(self, not_a_str: Any) -> None:
        assert not isinstance(not_a_str, str)

    def test_an_int(self, an_int: Any) -> None:
        assert is_integer(an_int)

    def test_not_an_int(self, not_an_int: Any) -> None:
        assert not is_integer(not_an_int)

    def test_a_float(self, a_float: Any) -> None:
        assert (
            is_numeric(a_float)
            and not is_integer(a_float)
        )

    def test_not_a_float(self, not_a_float: Any) -> None:
        assert (
            not is_numeric(not_a_float)
            or is_integer(not_a_float)
            or is_complex(not_a_float)
        )

    def test_a_complex(self, a_complex: Any) -> None:
        assert is_complex(a_complex)

    def test_a_bool(self, a_bool: Any) -> None:
        assert isinstance(a_bool, (bool, np.bool_))

    def test_not_a_bool(self, not_a_bool: Any) -> None:
        assert not isinstance(not_a_bool, (bool, np.bool_))

    def test_a_seq_str(self, a_seq_str: Any) -> None:
        assert is_sequence(a_seq_str)
        assert len(a_seq_str) >= 0
        assert all(isinstance(s, str) for s in a_seq_str)

    def test_not_a_seq_str(self, not_a_seq_str: Any) -> None:
        assert (
            not is_sequence(not_a_seq_str)
            or isinstance(not_a_seq_str, str)
            or any(not isinstance(s, str) for s in not_a_seq_str)
        )

    def test_a_seq_int(self, a_seq_int: Any) -> None:
        assert is_sequence(a_seq_int)
        assert len(a_seq_int) >= 0
        assert all(is_integer(i) for i in a_seq_int)

    def test_not_a_seq_int(self, not_a_seq_int: Any) -> None:
        assert (
            not is_sequence(not_a_seq_int)
            or isinstance(not_a_seq_int, str)
            or any(not is_integer(i) for i in not_a_seq_int)
        )

    def test_not_a_seq_float(self, not_a_seq_float: Any) -> None:
        assert (
            not is_sequence(not_a_seq_float)
            or isinstance(not_a_seq_float, str)
            or any(
                not is_numeric(f)
                or is_integer(f)
                or is_complex(f)
                for f in not_a_seq_float
            )
        )

    def test_a_seq_complex(self, a_seq_complex: Any) -> None:
        assert is_sequence(a_seq_complex)
        assert len(a_seq_complex) >= 0
        assert all(is_complex(c) for c in a_seq_complex)

    def test_a_seq_bool(self, a_seq_bool: Any) -> None:
        assert is_sequence(a_seq_bool)
        assert len(a_seq_bool) >= 0
        assert all(isinstance(b, (bool, np.bool_)) for b in a_seq_bool)

    def test_not_a_seq_bool(self, not_a_seq_bool: Any) -> None:
        assert (
            not is_sequence(not_a_seq_bool)
            or isinstance(not_a_seq_bool, str)
            or any(not isinstance(b, (bool, np.bool_)) for b in not_a_seq_bool)
        )
