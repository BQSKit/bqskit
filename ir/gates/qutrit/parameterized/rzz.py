"""This module implements the RZZGate."""
from __future__ import annotations

import jax
import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qutrit.constant.zz import Z0Z0Gate
from bqskit.ir.gates.qutrit.constant.zz import Z1Z1Gate
from bqskit.ir.gates.qutrit.constant.zz import Z2Z2Gate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RZ0Z0Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """A gate representing an arbitrary rotation around the ZZ axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rz0z0'

    zz = np.array(ZZGate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.zz)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        return UnitaryMatrix(np.array(self._unitary(params)))

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        return np.array(
            [
                np.array(jax.jacfwd(self._unitary)(params)),
            ], dtype=np.complex128,
        )


class RZ1Z1Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """A gate representing an arbitrary rotation around the ZZ axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rz1z1'

    zz = np.array(Z1Z1Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.zz)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        return UnitaryMatrix(np.array(self._unitary(params)))

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        return np.array(
            [
                np.array(jax.jacfwd(self._unitary)(params)),
            ], dtype=np.complex128,
        )


class RZ2Z2Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """A gate representing an arbitrary rotation around the ZZ axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rz2z2'

    zz = np.array(Z2Z2Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.zz)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        return UnitaryMatrix(np.array(self._unitary(params)))

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        return np.array(
            [
                np.array(jax.jacfwd(self._unitary)(params)),
            ], dtype=np.complex128,
        )
