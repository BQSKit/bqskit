"""This module tests the U1Gate class."""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from hypothesis import given
from hypothesis.strategies import floats, integers, lists

from bqskit.ir.gates.parameterized import MCRYGate, RYGate
from bqskit.ir.gates.constant import PermutationGate

@given(lists(elements=floats(allow_nan=False, allow_infinity=False, width=32), min_size=2, max_size=16))
def test_get_unitary(thetas: list[float]) -> None:
    '''
    Test the get_unitary method of the MCRYGate class.
    Use the default target qubit.
    '''
    # Assert that len(thetas) is a power of 2
    assert (len(thetas) & (len(thetas) - 1)) == 0

    # There are 2 ** (n - 1) parameters
    num_qudits = np.log2(len(thetas)) + 1
    mcry = MCRYGate(num_qudits=num_qudits)
    block_unitaries = [RYGate().get_unitary([theta]) for theta in thetas]
    blocked_unitary = block_diag(*block_unitaries)
    dist = mcry.get_unitary(thetas).get_distance_from(blocked_unitary)
    assert dist < 1e-7

@given(integers(min_value=0, max_value=4))
def test_get_unitary_target_select(target_qubit: int) -> None:
    '''
    Test the get_unitary method of the MCRYGate class when
    the target qubit is set.
    '''
    # Create an MCRY gate with 6 qubits and random parameters
    mcry = MCRYGate(num_qudits=6, target_qubit=target_qubit)
    thetas = list(np.random.rand(2 ** 5) * 2 * np.pi)

    # Create the block diagonal matrix
    block_unitaries = [RYGate().get_unitary([theta]) for theta in thetas]
    blocked_unitary = block_diag(*block_unitaries)

    # Apply a permutation to the block diagonal matrix
    # Swap the target qubit with the last qubit
    perm = np.arange(6)
    perm[-1], perm[target_qubit] = perm[target_qubit], perm[-1]
    perm_gate = PermutationGate(6, perm) 

    full_utry = perm_gate.get_unitary() @ blocked_unitary

    dist = mcry.get_unitary(thetas).get_distance_from(full_utry)
    assert dist < 1e-7