"""This module implements the MCRYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
import logging

def get_indices(index: int, target_qudit, num_qudits):
    """
    Get indices for the matrix based on the target qubit.
    """
    shift_qubit = num_qudits - target_qudit - 1
    shift = 2 ** shift_qubit
    # Split into two parts around target qubit
    # 100 | 111
    left = index // shift
    right = index % shift

    # Now, shift left by one spot to 
    # make room for the target qubit
    left *= (shift * 2)
    # Now add 0 * new_ind and 1 * new_ind to get indices
    return left + right, left + shift + right

class MCRYGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
    LocallyOptimizableUnitary
):
    """
    A gate representing a multiplexed Y rotation. A multiplexed Y rotation
    uses n - 1 qubits as select qubits and applies a Y rotation to the target.
    If the target qubit is the last qubit, then the unitary is block diagonal.
    Each block is a 2x2 RY matrix with parameter theta. 

    Since there are n - 1 select qubits, there are 2^(n-1) parameters (thetas).

    We allow the target qubit to be specified to any qubit, and the other qubits
    maintain their order. Qubit 0 is the most significant qubit. 

    See this paper: https://arxiv.org/pdf/quant-ph/0406176
    """

    _qasm_name = 'mcry'

    def __init__(self, num_qudits: int, controlled_qubit: int) -> None:
        self._num_qudits = num_qudits
        # 1 param for each configuration of the selec qubits
        self._num_params = 2 ** (num_qudits - 1)
        self.controlled_qubit = controlled_qubit
        super().__init__()

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        matrix = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for i, param in enumerate(params):
            cos = np.cos(param / 2)
            sin = np.sin(param / 2)

            # Now, get indices based on target qubit.
            # i corresponds to the configuration of the 
            # select qubits (e.g 5 = 101). Now, the 
            # target qubit is 0,1 for both the row and col
            # indices. So, if i = 5 and the target_qubit is 2
            # Then the rows/cols are 1001 and 1101
            x1, x2 = get_indices(i, self.controlled_qubit, self.num_qudits)

            matrix[x1, x1] = cos
            matrix[x2, x2] = cos
            matrix[x2, x1] = sin
            matrix[x1, x2] = -1 * sin

        return UnitaryMatrix(matrix)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.
        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        matrix = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for i, param in enumerate(params):
            dcos = -np.sin(param / 2) / 2
            dsin = -1j * np.cos(param / 2) / 2

            # Again, get indices based on target qubit.
            x1, x2 = get_indices(i, self.controlled_qubit, self.num_qudits)

            matrix[x1, x1] = dcos
            matrix[x2, x2] = dcos
            matrix[x2, x1] = dsin
            matrix[x1, x2] = -1 * dsin

        return UnitaryMatrix(matrix)


    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.
        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        thetas = [0] * self.num_params

        for i in range(self.num_params):
            x1, x2 = get_indices(i, self.controlled_qubit, self.num_qudits)
            a = np.real(env_matrix[x1, x1] + env_matrix[x2, x2])
            b = np.real(env_matrix[x2, x1] - env_matrix[x1, x2])
            theta = 2 * np.arccos(a / np.sqrt(a ** 2 + b ** 2))
            theta *= -1 if b > 0 else 1
            thetas[i] = theta

        return thetas

    @staticmethod
    def get_decomposition(params: RealVector = []) -> tuple[RealVector, RealVector]:
        '''
        Get the corresponding parameters for one level of decomposition
        of a multiplexed gate. This is used in the decomposition of both
        the MCRY and MCRZ gates. See :class:`MGDPass` for more info.
        '''
        new_num_params = len(params) // 2
        left_params = np.zeros(new_num_params)
        right_params = np.zeros(new_num_params)
        for i in range(len(left_params)):
            left_param = (params[i] + params[i + new_num_params]) / 2
            right_param = (params[i] - params[i + new_num_params]) / 2
            left_params[i] = left_param
            right_params[i] = right_param

        return left_params, right_params

    @property
    def name(self) -> str:
        """The name of this gate, with the number of qudits appended."""
        base_name = getattr(self, '_name', self.__class__.__name__)
        return f"{base_name}_{self.num_qudits}"