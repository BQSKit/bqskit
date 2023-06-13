"""This module implements the CRZGate."""
from __future__ import annotations

import jax
import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.ir.gates.qutrit import jaxcgate
from bqskit.ir.gates.qutrit import pdict
from bqskit.ir.gates.qutrit.constant.z import Z0Gate
from bqskit.ir.gates.qutrit.constant.z import Z1Gate
from bqskit.ir.gates.qutrit.constant.z import Z2Gate
from bqskit.ir.gates.qutrit.constant.z import ZGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
jax.config.update('jax_enable_x64', True)


class CRZ0Gate():
    """A gate representing a controlled rotation around the z-axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 2
    _qasm_name = 'crz0'

    z = np.array(Z0Gate._utry)

    def _unitary(self, params):
        return jaxcgate(params[0], jax.scipy.linalg.expm(-1j * params[1] * self.z))

    def _unitary_diff(self, params):
        return jax.numpy.kron(pdict[params[0]], jax.scipy.linalg.expm(-1j * params[1] * self.z))

    def get_unitary(self, params):
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self._unitary(params)

    def get_grad(self, params) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        def _temp(param):
            return self._unitary_diff([params[0], param])

        return np.array(
            [
                np.zeros((9, 9)),
                   np.array(jax.jacfwd(_temp)(params[1])),
            ], dtype=np.complex128,
        )


class CRZ1Gate():
    """A gate representing a controlled rotation around the z-axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 2
    _qasm_name = 'crz1'

    z = np.array(Z1Gate._utry)

    def _unitary(self, params):
        return jaxcgate(params[0], jax.scipy.linalg.expm(-1j * params[1] * self.z))

    def _unitary_diff(self, params):
        return jax.numpy.kron(pdict[params[0]], jax.scipy.linalg.expm(-1j * params[1] * self.z))

    def get_unitary(self, params):
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self._unitary(params)

    def get_grad(self, params) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        def _temp(param):
            return self._unitary_diff([params[0], param])

        return np.array(
            [
                np.zeros((9, 9)),
                   np.array(jax.jacfwd(_temp)(params[1])),
            ], dtype=np.complex128,
        )


class CRZ2Gate():
    """A gate representing a controlled rotation around the z-axis for
    qutrits."""

    _num_qudits = 2
    _num_params = 2
    _qasm_name = 'crz2'

    z = np.array(Z2Gate._utry)

    def _unitary(self, params):
        return jaxcgate(params[0], jax.scipy.linalg.expm(-1j * params[1] * self.z))

    def _unitary_diff(self, params):
        return jax.numpy.kron(pdict[params[0]], jax.scipy.linalg.expm(-1j * params[1] * self.z))

    def get_unitary(self, params):
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self._unitary(params)

    def get_grad(self, params) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        def _temp(param):
            return self._unitary_diff([params[0], param])

        return np.array(
            [
                np.zeros((9, 9)),
                   np.array(jax.jacfwd(_temp)(params[1])),
            ], dtype=np.complex128,
        )
