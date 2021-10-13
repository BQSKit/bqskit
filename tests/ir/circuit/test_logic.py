"""This module tests circuit.renumber_qudits."""
from __future__ import annotations

from hypothesis import given

from bqskit.ir.circuit import Circuit
from bqskit.qis.permutation import calc_permutation_matrix
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.test.strategies import circuits
from bqskit.utils.test.types import invalid_type_test
from bqskit.utils.test.types import valid_type_test


@given(circuits((2, 2, 2)))
def test_get_inverse(circuit: Circuit) -> None:
    U = circuit.get_unitary()
    UT = circuit.get_inverse().get_unitary()
    assert U @ UT == UnitaryMatrix.identity(U.dim, U.radixes)
    assert UT @ U == UnitaryMatrix.identity(U.dim, U.radixes)


@valid_type_test(Circuit(1).renumber_qudits)
def test_valid_type() -> None:
    pass


@invalid_type_test(Circuit(1).renumber_qudits)
def test_invalid_type() -> None:
    pass


@given(circuits((2, 2, 2)))
def test_renumber(circuit: Circuit) -> None:
    U = circuit.get_unitary()
    P = calc_permutation_matrix(3, [1, 2, 0])
    circuit.renumber_qudits([1, 2, 0])
    U2 = circuit.get_unitary()
    assert U == P.T @ U2 @ P
