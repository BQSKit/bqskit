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

_logger = logging.getLogger(__name__)

def get_indices(index: int, controlled_qudit, num_qudits):
    # 0 corresponds to MSB
    shift_qubit = num_qudits - controlled_qudit - 1
    shift = 2 ** shift_qubit
    # Split into two parts around controlled qubit
    # 100 | 111
    left = index // shift
    right = index % shift
    # Now, shift left
    left *= (shift * 2)
    # Now add 0 * new_ind and 1 * new_ind to ge indices
    return left + right, left + shift + right

class MCRYGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
    LocallyOptimizableUnitary
):
    """
    A gate representing a multi-controlled Y rotation.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}} \\\\
        0 & 0 & \\sin{\\frac{\\theta}{2}} & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
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
        if len(params) == 1:
            # If we want to turn on only if all the selects are 1, that corresponds to
            # [0, 0, ...., 0, theta] so that is why we pad in front
            params = list(np.zeros(len(params) - self.num_params)) + list(params)
        self.check_parameters(params)

        matrix = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for i, param in enumerate(params):
            cos = np.cos(param / 2)
            sin = np.sin(param / 2)

            # Now, get indices based on control qubit.
            # i corresponds to the configuration of the 
            # select qubits (e.g 5 = 101). Now, the 
            # controlled qubit is 0,1 for both the row and col
            # indices. So, if i = 5 and the controlled_qubit is 2
            # Then the rows/cols are 1001 and 1101
            # Use helper function
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
        if len(params) < self.num_params:
            # Pad Zeros to front
            params = list(np.zeros(len(params) - self.num_params)) + list(params)
        self.check_parameters(params)

        matrix = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for i, param in enumerate(params):
            dcos = -np.sin(param / 2) / 2
            dsin = -1j * np.cos(param / 2) / 2

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
        print("Using python optimize :(")
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
    def get_decomposition(params: RealVector = []) -> tuple[RealVector, RealVector] :
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
        """The name of this gate."""
        base_name = getattr(self, '_name', self.__class__.__name__)
        return f"{base_name}_{self.num_qudits}"