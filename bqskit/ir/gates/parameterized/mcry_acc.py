"""This module implements the MCRYGate."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import jax.random as rand
import jax.scipy.linalg as jla
import numpy.typing as npt
import jax.typing as jpt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskitqfactorjax.unitarymatrixjax import UnitaryMatrixJax
from bqskit.utils.cachedclass import CachedClass
from typing import Sequence
import logging

_logger = logging.getLogger(__name__)

class MCRYAccGate(
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

    def __init__(self, num_qudits: int) -> None:
        self._num_qudits = num_qudits
        # 1 param for each configuration of the selec qubits
        self._num_params = 2 ** (num_qudits - 1)
        super().__init__()

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrixJax:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if len(params) == 0: # Random parameters
            params = rand.uniform(rand.PRNGKey(23), shape=(self.num_params,))

        if len(params) == 1:
            # If we want to turn on only if all the selects are 1, that corresponds to
            # [0, 0, ...., 0, theta] so that is why we pad in front
            params = list(jnp.zeros(len(params) - self.num_params)) + list(params)

        params = jnp.array(params)
        cos = jnp.cos(params / 2)
        sin = jnp.sin(params / 2)
        zeros = jnp.zeros(len(params))
        doub_cos = jnp.vstack([cos, cos]).flatten(order='F') # double cos vector
        sin_diag = jnp.vstack([sin, zeros]).flatten(order='F') # double sin vector
        doub_cos = jnp.array(doub_cos, dtype=jnp.complex128)
        sin_diag = jnp.array(sin_diag, dtype=jnp.complex128)
        negsin_diag = -1 * sin_diag
        cos_arr = jnp.diag(doub_cos)
        sin_arr = jnp.diag(sin_diag[:-1], k=1)
        negsin_arr = jnp.diag(negsin_diag[:-1], k=-1)
        matrix = cos_arr + sin_arr + negsin_arr

        return UnitaryMatrixJax(matrix)

    def get_grad(self, params: RealVector = []) -> jpt.NDArray[jnp.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if len(params) == 0:
            # params = rand.uniform(rand.PRNGKey(23), shape=(self.num_params,))
            params = jnp.zeros((len(params),)) 

        if len(params) < self.num_params:
            # Pad Zeros to front
            params = list(jnp.zeros(len(params) - self.num_params)) + list(params)

        params = jnp.array(params)
        dcos = -jnp.sin(params / 2) / 2
        dsin = -1j * jnp.cos(params / 2) / 2
        zeros = jnp.zeros(len(params))
        doub_cos = jnp.vstack([dcos, dcos]).flatten(order='F') # double cos vector
        sin_diag = jnp.vstack([dsin, zeros]).flatten(order='F') # double sin vector
        doub_cos = jnp.array(doub_cos, dtype=jnp.complex128)
        sin_diag = jnp.array(sin_diag, dtype=jnp.complex128)
        negdsin_diag = -1 * sin_diag
        cos_arr = jnp.diag(doub_cos)
        sin_arr = jnp.diag(sin_diag[:-1], k=1)
        negsin_arr = jnp.diag(negdsin_diag[:-1], k=-1)
        matrix = cos_arr + sin_arr + negsin_arr

        return UnitaryMatrixJax(matrix)


    def optimize(self, env_matrix: jpt.NDArray[jnp.complex128], get_untry=False, prev_utry: UnitaryMatrixJax = None, beta: float = 0.0) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        thetas = [0] * self.num_params
        diag = jnp.diag(env_matrix)
        up_diag = jnp.diag(env_matrix, k = 1)
        down_diag = jnp.diag(env_matrix, k = -1)
        a = jnp.real(diag[0::2] + diag[1::2])
        b = jnp.real(down_diag - up_diag)[0::2]
        thetas = 2 * jnp.arccos(a / jnp.sqrt(a ** 2 + b ** 2))
        if get_untry:
            return self.get_unitary(thetas)
        return thetas
    
    @property
    def name(self) -> str:
        """The name of this gate."""
        base_name = getattr(self, '_name', self.__class__.__name__)
        return f"{base_name}_{self.num_qudits}"
    
    @staticmethod
    def get_params(utry: jpt.NDArray[jnp.complex128]) -> RealVector:
        """Return the params for this gate, given a unitary matrix."""
        diag = jnp.diag(utry)
        a = jnp.arccos(diag[0::2]) * 2
        return list(np.array(a, dtype=np.float64))