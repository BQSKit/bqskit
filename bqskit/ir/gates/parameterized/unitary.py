"""This module implements the VariableUnitaryGate."""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy.typing as npt
import scipy as sp

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_valid_radixes


class VariableUnitaryGate(
    Gate,
    LocallyOptimizableUnitary,
):
    """A Variable n-qudit unitary operator."""

    def __init__(self, num_qudits: int, radixes: Sequence[int] = []) -> None:
        """
        Creates an VariableUnitaryGate, defaulting to a qubit gate.

        Args:
            num_qudits (int): The number of qudits this gate acts on.

            radixes (Sequence[int]): The number of orthogonal
                states for each qudit. Defaults to qubits.
        """
        if num_qudits <= 0:
            raise ValueError('Expected positive integer, got %d' % num_qudits)

        if len(radixes) == 0:
            radixes = [2] * num_qudits

        if not is_valid_radixes(radixes, num_qudits):
            raise TypeError('Invalid radixes.')

        self._num_qudits = int(num_qudits)
        self._radixes = tuple(radixes)
        self._dim = int(np.prod(self.radixes))
        self.shape = (self.dim, self.dim)
        self._num_params = 2 * self.dim**2
        self._name = 'VariableUnitaryGate(%d, %s)' % (
            self.num_qudits, str(self.radixes),
        )

    def get_unitary(self, params: RealVector = [], check_params: bool = True, use_jax: bool = False) -> UnitaryMatrix:
        """
        Return the unitary for this gate, see :class:`Unitary` for more.

        Note:
            Ideally, params form a unitary matrix when reshaped,
            however, params are unconstrained so we return the closest
            UnitaryMatrix to the given matrix.
        """
        if check_params:
            self.check_parameters(params)
        mid = len(params) // 2
        if not use_jax:
            mat_lib = np
        else:
            mat_lib = jnp
        real = mat_lib.array(params[:mid], dtype=mat_lib.complex128)
        imag = 1j * mat_lib.array(params[mid:], dtype=mat_lib.complex128)
        x = real + imag
        return UnitaryMatrix.closest_to(mat_lib.reshape(x, self.shape), self.radixes)

    def optimize(self, env_matrix: npt.NDArray[np.complex128], get_untry:bool = False, use_jax: bool = False) -> Union[list[float], UnitaryMatrix]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        if not use_jax:
            U, _, Vh = sp.linalg.svd(env_matrix)
            mat_lib = np
        else:
            U, _, Vh = jla.svd(env_matrix)
            mat_lib = jnp
        utry = Vh.conj().T @ U.conj().T
        
        if get_untry:
            return UnitaryMatrix(utry, radixes=self._radixes, check_arguments=False)
        
        x = mat_lib.reshape(utry, (self.num_params // 2,))
        return list(mat_lib.real(x)) + list(mat_lib.imag(x))

    @staticmethod
    def get_params(utry: UnitaryLike, use_jax: bool = False) -> RealVector:
        """Return the params for this gate, given a unitary matrix."""
        num_elems = len(utry) ** 2
        if not use_jax:
            mat_lib = np
        else:
            mat_lib = jnp
        real = mat_lib.reshape(mat_lib.real(utry), num_elems)
        imag = mat_lib.reshape(mat_lib.imag(utry), num_elems)
        return mat_lib.concatenate([real, imag])

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, VariableUnitaryGate)
            and self.num_qudits == other.num_qudits
            and self.radixes == other.radixes
        )

    def __hash__(self) -> int:
        return hash((self.num_qudits, self.radixes))
