"""This module implements the MCRZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.ir.gates.parameterized.mcry import get_indices
from typing import Any

class MCRZGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
    LocallyOptimizableUnitary
):
    """
    A gate representing a multiplexed Z rotation. A multiplexed Z rotation
    uses n - 1 qubits as select qubits and applies a Z rotation to the target.
    If the target qubit is the last qubit, then the unitary is block diagonal.
    Each block is a 2x2 RZ matrix with parameter theta. 

    Since there are n - 1 select qubits, there are 2^(n-1) parameters (thetas).

    We allow the target qubit to be specified to any qubit, and the other qubits
    maintain their order. Qubit 0 is the most significant qubit. 


    Why is 0 the MSB? Typically, in the QSD diagram, we see the block drawn
    with qubit 0 at the top and qubit n-1 at the bottom. Then, the decomposition
    slowly moves from the bottom to the top.

    See this paper: https://arxiv.org/pdf/quant-ph/0406176
    """

    _qasm_name = 'mcrz'

    def __init__(self, num_qudits: int, target_qubit: int = -1) -> None:
        '''
        Create a new MCRZGate with `num_qudits` qubits and 
        `target_qubit` as the target qubit. We then have 2^(n-1) parameters
        for this gate.

        For Example:
        `num_qudits` = 3, `target_qubit` = 1

        Then, the select qubits are 0 and 2 with 0 as the MSB.

        If the input vector is |0x0> then the selection is 00, and
        RZ(theta_0) is applied to the target qubit.

        If the input vector is |1x0> then the selection is 01, and
        RZ(theta_1) is applied to the target qubit.
        '''
        self._num_qudits = num_qudits
        # 1 param for each configuration of the selec qubits
        self._num_params = 2 ** (num_qudits - 1)
        # By default, the controlled qubit is the last qubit
        if target_qubit == -1:
            target_qubit = num_qudits - 1
        self.target_qubit = target_qubit
        super().__init__()

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        matrix = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for i, param in enumerate(params):
            pos = np.exp(1j * param / 2)
            neg = np.exp(-1j * param / 2)

            # Get correct indices based on target qubit
            # See :class:`mcry` for more info
            x1, x2 = get_indices(i, self.target_qubit, self.num_qudits)

            matrix[x1, x1] = neg
            matrix[x2, x2] = pos

        return UnitaryMatrix(matrix)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)
        matrix = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for i, param in enumerate(params):
            dpos = 1j / 2 * np.exp(1j * param / 2)
            dneg = -1j / 2 * np.exp(-1j * param / 2)

            # Again, get indices based on target qubit.
            x1, x2 = get_indices(i, self.target_qubit, self.num_qudits)

            matrix[x1, x1] = dpos
            matrix[x2, x2] = dneg

        return UnitaryMatrix(matrix)


    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.
        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        thetas = [0] * self.num_params

        for i in range(self.num_params):
            x1, x2 = get_indices(i, self.target_qubit, self.num_qudits)
            # Optimize each RZ independently from indices
            # Taken from QFACTOR repo
            a = np.angle(env_matrix[x1, x1])
            b = np.angle(env_matrix[x2, x2])
            # print(thetas)
            thetas[i] = a - b

        return thetas

    @property
    def name(self) -> str:
        """The name of this gate, with the number of qudits appended."""
        base_name = getattr(self, '_name', self.__class__.__name__)
        return f"{base_name}_{self.num_qudits}"