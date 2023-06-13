"""This module implements the RZGate."""
from __future__ import annotations

import jax
import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qutrit.constant.z import Z0Gate
from bqskit.ir.gates.qutrit.constant.z import Z1Gate
from bqskit.ir.gates.qutrit.constant.z import Z2Gate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
jax.config.update('jax_enable_x64', True)


class RZ0Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """A gate representing an arbitrary rotation around the Z axis for
    qutrits."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz0'

    z = np.array(Z0Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.z)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return UnitaryMatrix(self._unitary(params))

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


class RZ1Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """A gate representing an arbitrary rotation around the Z axis for
    qutrits."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz1'

    z = np.array(Z1Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.z)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return UnitaryMatrix(self._unitary(params))

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


class RZ2Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """A gate representing an arbitrary rotation around the Z axis for
    qutrits."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz2'

    z = np.array(Z2Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.z)

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return UnitaryMatrix(self._unitary(params))

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
