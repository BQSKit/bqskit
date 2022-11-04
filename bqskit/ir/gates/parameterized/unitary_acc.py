from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import Array

from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary.unitarymatrixjax import UnitaryMatrixJax


class VariableUnitaryGateAcc(VariableUnitaryGate):
    """A Variable n-qudit unitary operator, that uses JAX as the math
    library."""

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """
        Return the unitary for this gate, see :class:`Unitary` for more.

        Note:
            Ideally, params form a unitary matrix when reshaped,
            however, params are unconstrained so we return the closest
            UnitaryMatrix to the given matrix.
        """
        mid = len(params) // 2
        real = jnp.array(params[:mid], dtype=jnp.complex128)
        imag = 1j * jnp.array(params[mid:], dtype=jnp.complex128)
        x = real + imag
        return UnitaryMatrixJax.closest_to(jnp.reshape(x, self.shape), self.radixes)

    def optimize(self, env_matrix, get_untry: bool = False) -> list[float] | UnitaryMatrixJax:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """

        U, _, Vh = jla.svd(env_matrix)
        utry = Vh.conj().T @ U.conj().T

        if get_untry:
            return UnitaryMatrixJax(utry, radixes=self._radixes)

        x = jnp.reshape(utry, (self.num_params // 2,))
        return list(jnp.real(x)) + list(jnp.imag(x))

    @staticmethod
    def get_params(utry: UnitaryLike) -> RealVector:
        """Return the params for this gate, given a unitary matrix."""
        num_elems = len(utry) ** 2
        real = jnp.reshape(jnp.real(utry), num_elems)
        imag = jnp.reshape(jnp.imag(utry), num_elems)
        return jnp.concatenate([real, imag])
