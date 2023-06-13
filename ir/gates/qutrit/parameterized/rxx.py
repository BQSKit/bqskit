"""This module implements the RXXGate."""
from __future__ import annotations

import jax
import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qutrit.constant.xx import X01X01Gate
from bqskit.ir.gates.qutrit.constant.xx import X01X02Gate
from bqskit.ir.gates.qutrit.constant.xx import X02X01Gate
from bqskit.ir.gates.qutrit.constant.xx import X02X02Gate
from bqskit.ir.gates.qutrit.constant.xx import XXGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RX01X01Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """A gate representing an arbitrary rotation around the XX axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rx01x01'

    xx = np.array(X01X01Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.xx)

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


class RX02X02Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """A gate representing an arbitrary rotation around the XX axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rx02x02'

    xx = np.array(X02X02Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.xx)

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


class RX01X02Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """A gate representing an arbitrary rotation around the XX axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rx01x02'

    xx = np.array(X01X02Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.xx)

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


class RX02X01Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """A gate representing an arbitrary rotation around the XX axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rx02x01'

    xx = np.array(X02X01Gate._utry)

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j * params[0] * self.xx)

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
