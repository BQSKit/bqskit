"""This module implements the PauliGate."""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import scipy as sp

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.pauli import PauliMatrices
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.math import dexpmv
from bqskit.utils.math import dot_product
from bqskit.utils.math import pauli_expansion
from bqskit.utils.math import unitary_log_no_i


class PauliGate(QubitGate, DifferentiableUnitary, LocallyOptimizableUnitary):
    """
    A gate representing an arbitrary rotation.

    This gate is given by:

    .. math::

        \\exp({i(\\vec{\\alpha} \\cdot \\vec{\\sigma^{\\otimes n}})})

    Where :math:`\\vec{\\alpha}` are the gate's parameters,
    :math:`\\vec{\\sigma}` are the Pauli matrices,
    and :math:`n` is the number of qubits this gate acts on.
    """

    def __init__(self, num_qudits: int) -> None:
        """
        Create a PauliGate acting on `num_qudits` qubits.

        Args:
            num_qudits (int): The number of qudits this gate will act on.

        Raises:
            ValueError: If `num_qudits` is nonpositive.
        """

        if num_qudits <= 0:
            raise ValueError('Expected positive integer, got %d' % num_qudits)

        self._num_qudits = num_qudits
        self.paulis = PauliMatrices(self.num_qudits)
        self._num_params = len(self.paulis)
        if 'READTHEDOCS' in os.environ:
            self.sigmav = None
        else:
            self.sigmav = (-1j / 2) * self.paulis.numpy

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        H = dot_product(params, self.sigmav)
        eiH = sp.linalg.expm(H)
        return UnitaryMatrix(eiH, check_arguments=False)

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        H = dot_product(params, self.sigmav)
        _, dU = dexpmv(H, self.sigmav)
        return dU

    def get_unitary_and_grad(
        self,
        params: Sequence[float] = [],
    ) -> tuple[UnitaryMatrix, np.ndarray]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        H = dot_product(params, self.sigmav)
        U, dU = dexpmv(H, self.sigmav)
        return UnitaryMatrix(U, check_arguments=False), dU

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        U, _, Vh = sp.linalg.svd(env_matrix)
        utry = Vh.conj().T @ U.conj().T
        return list(-2 * pauli_expansion(unitary_log_no_i(utry)))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PauliGate) and self.num_qudits == o.num_qudits

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.num_qudits))
