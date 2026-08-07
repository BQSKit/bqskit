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
"""
from __future__ import annotations

from typing import Any

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary import UnitaryMatrix


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
