"""This module tests the MPRYGate class."""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import lists

from bqskit.ir.gates.constant import PermutationGate
from bqskit.ir.gates.parameterized import MPRYGate
from bqskit.ir.gates.parameterized import RYGate


@given(
    lists(
        elements=floats(
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ), min_size=2, max_size=16,
    ),
)
def test_get_unitary(thetas: list[float]) -> None:
    """
    Test the get_unitary method of the MPRYGate class.

    Use the default target qubit.
    """
    # Ensure that len(thetas) is a power of 2
    # There are 2 ** (n - 1) parameters
    num_qudits = int(np.log2(len(thetas))) + 1
    thetas = thetas[:2 ** (num_qudits - 1)]
    MPRy = MPRYGate(num_qudits=num_qudits)
    block_unitaries = [RYGate().get_unitary([theta]) for theta in thetas]
    blocked_unitary = la.block_diag(*block_unitaries)
    dist = MPRy.get_unitary(thetas).get_distance_from(blocked_unitary)
    assert dist < 1e-7


@given(integers(min_value=0, max_value=4))
def test_get_unitary_target_select(target_qubit: int) -> None:
    """Test the get_unitary method of the MPRYGate class when the target qubit
    is set."""
    # Create an MPRY gate with 6 qubits and random parameters
    num_qudits = 6
    MPRy = MPRYGate(num_qudits=num_qudits, target_qubit=target_qubit)
    thetas = list(np.random.rand(2 ** (num_qudits - 1)) * 2 * np.pi)

    # Create the block diagonal matrix
    block_unitaries = [RYGate().get_unitary([theta]) for theta in thetas]
    blocked_unitary = la.block_diag(*block_unitaries)

    # Apply a permutation transformation
    # to the block diagonal matrix
    # Swap the target qubit with the last qubit
    # perm = np.arange(num_qudits)
    perm = list(range(num_qudits))
    for i in range(target_qubit, num_qudits):
        perm[i] = i + 1
    perm[-1] = target_qubit

    perm_gate = PermutationGate(num_qudits, perm)

    full_utry = (
        perm_gate.get_unitary().conj().T
        @ blocked_unitary @ perm_gate.get_unitary()
    )

    dist = MPRy.get_unitary(thetas).get_distance_from(full_utry)
    assert dist < 1e-7
