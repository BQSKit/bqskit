import numpy as np
from scipy.linalg import block_diag

from bqskit.ir.gates.parameterized import MCRYGate, RYGate, MCRZGate, RZGate
from bqskit.ir.gates.constant import PermutationGate

def test_get_unitary_mcry(thetas: list[float]) -> None:
    '''
    Test the get_unitary method of the MCRYGate class.
    Use the default target qubit.
    '''
    # Ensure that len(thetas) is a power of 2
    # There are 2 ** (n - 1) parameters
    num_qudits = int(np.log2(len(thetas))) + 1
    thetas = thetas[:2 ** (num_qudits - 1)]

    mcry = MCRYGate(num_qudits=num_qudits)
    block_unitaries = [RYGate().get_unitary([theta]) for theta in thetas]
    blocked_unitary = block_diag(*block_unitaries)
    dist = mcry.get_unitary(thetas).get_distance_from(blocked_unitary)
    assert dist < 1e-7

def test_get_unitary_mcrz(thetas: list[float]) -> None:
    '''
    Test the get_unitary method of the MCRYGate class.
    Use the default target qubit.
    '''
    # Ensure that len(thetas) is a power of 2
    # There are 2 ** (n - 1) parameters
    num_qudits = int(np.log2(len(thetas))) + 1
    thetas = thetas[:2 ** (num_qudits - 1)]

    mcry = MCRZGate(num_qudits=num_qudits)
    block_unitaries = [RZGate().get_unitary([theta]) for theta in thetas]
    blocked_unitary = block_diag(*block_unitaries)
    dist = mcry.get_unitary(thetas).get_distance_from(blocked_unitary)
    assert dist < 1e-7

def test_get_unitary_target_select_mcry(target_qubit: int) -> None:
    '''
    Test the get_unitary method of the MCRYGate class when
    the target qubit is set.
    '''
    # Create an MCRY gate with 6 qubits and random parameters
    num_qudits = 6
    mcry = MCRYGate(num_qudits=num_qudits, target_qubit=target_qubit)
    thetas = list(np.random.rand(2 ** (num_qudits - 1)) * 2 * np.pi)

    # Create the block diagonal matrix
    block_unitaries = [RYGate().get_unitary([theta]) for theta in thetas]
    blocked_unitary = block_diag(*block_unitaries)

    # Apply a permutation transformation 
    # to the block diagonal matrix
    # Swap the target qubit with the last qubit
    # perm = np.arange(num_qudits)
    perm = list(range(num_qudits))
    for i in range(target_qubit, num_qudits):
        perm[i] = i + 1
    perm[-1] = target_qubit

    perm_gate = PermutationGate(num_qudits, perm) 

    full_utry = perm_gate.get_unitary().conj().T @ blocked_unitary @ perm_gate.get_unitary()

    dist = mcry.get_unitary(thetas).get_distance_from(full_utry)
    assert dist < 1e-7




for num_params in [2,4,8, 20]:
    params = np.random.rand(num_params) * 2 * np.pi
    test_get_unitary_mcry(params)
    test_get_unitary_mcrz(params)

np.printoptions(precision=3, threshold=np.inf, linewidth=np.inf)

for target_qubit in [0,1,2,3,4,5]:
    test_get_unitary_target_select_mcry(target_qubit)