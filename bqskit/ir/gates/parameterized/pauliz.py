"""This module implements the PauliZGate."""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.pauliz import PauliZMatrices
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.math import dexpmv
from bqskit.utils.math import dot_product
from bqskit.utils.math import pauliz_expansion
from bqskit.utils.math import unitary_log_no_i


class PauliZGate(QubitGate, DifferentiableUnitary, GeneralGate):
    """
    A gate representing an arbitrary diagonal rotation.

    This gate is given by:

    .. math::

        \\exp({i(\\vec{\\alpha} \\cdot \\vec{\\sigma_Z^{\\otimes n}})})

    Where :math:`\\vec{\\alpha}` are the gate's parameters,
    :math:`\\vec{\\sigma}` are the PauliZ Z matrices,
    and :math:`n` is the number of qubits this gate acts on.
    """

    def __init__(self, num_qudits: int) -> None:
        """
        Create a PauliZGate acting on `num_qudits` qubits.

        Args:
            num_qudits (int): The number of qudits this gate will act on.

        Raises:
            ValueError: If `num_qudits` is nonpositive.
        """

        if num_qudits <= 0:
            raise ValueError(f'Expected positive integer, got {num_qudits}')

        self._name = f'PauliZGate({num_qudits})'
        self._num_qudits = num_qudits
        paulizs = PauliZMatrices(self.num_qudits)
        self._num_params = len(paulizs)
        if building_docs():
            self.sigmav: npt.NDArray[Any] = np.array([])
        else:
            self.sigmav = (-1j / 2) * paulizs.numpy

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        H = dot_product(params, self.sigmav)
        eiH = np.diag(np.exp(np.diag(H)))
        return UnitaryMatrix(eiH, check_arguments=False)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.

        TODO: Accelerated gradient computation for diagonal matrices.
        """
        self.check_parameters(params)
        H = dot_product(params, self.sigmav)
        _, dU = dexpmv(H, self.sigmav)
        return dU

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        H = dot_product(params, self.sigmav)
        U, dU = dexpmv(H, self.sigmav)
        return UnitaryMatrix(U, check_arguments=False), dU

    def calc_params(self, utry: UnitaryMatrix) -> list[float]:
        """Return the parameters for this gate to implement `utry`"""
        return list(-2 * pauliz_expansion(unitary_log_no_i(utry.numpy)))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PauliZGate) and self.num_qudits == o.num_qudits

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.num_qudits))
