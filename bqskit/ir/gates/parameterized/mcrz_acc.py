"""This module implements the MCRZGate."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import jax.random as rand
import jax.typing as jpt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskitqfactorjax.unitarymatrixjax import UnitaryMatrixJax
from bqskit.utils.cachedclass import CachedClass

class MCRZAccGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
    LocallyOptimizableUnitary
):
    """
    A gate representing a multiplexed Z rotation.

    It is given by the following parameterized unitary:
    """
    _qasm_name = 'mcrz'

    def __init__(self, num_qudits: int) -> None:
        # The bottom qubit is controlled
        self._num_qudits = num_qudits
        # 1 param for each configuration of the selec qubits
        self._num_params = 2 ** (num_qudits - 1)
        super().__init__()

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrixJax:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if len(params) == 0:
            # params = rand.uniform(rand.PRNGKey(23), shape=(self.num_params,))
            params = jnp.zeros((len(params),)) 
        
        if len(params) == 1:
            # If we want to turn on only if all the selects are 1, that corresponds to
            # [0, 0, ...., 0, theta] so that is why we pad in front
            params = list(jnp.zeros(self.num_params - 1)) + list(params)
        
        params = jnp.array(params)
        try:
            pos = jnp.exp(1j * params / 2)
        except:
            print(params)
            pos = jnp.exp(1j * params / 2)
        neg = jnp.exp(-1j * params / 2)
        z_diag = jnp.vstack((pos, neg)).flatten(order="F")
        z_diag = jnp.array(z_diag, dtype=jnp.complex128)

        return UnitaryMatrixJax(jnp.diag(z_diag))

    def get_grad(self, params: RealVector = []) -> jpt.NDArray[jnp.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if len(params) == 0:
            params = rand.uniform(rand.PRNGKey(23), shape=(self.num_params,))

        if len(params) < self.num_params:
            # Pad Zeros to front
            params = list(jnp.zeros(len(params) - self.num_params)) + list(params)

        dpos = 1j / 2 * jnp.exp(1j * params / 2)
        dneg = -1j / 2 * jnp.exp(-1j * params / 2)
        z_diag = jnp.vstack([dpos, dneg]).flatten(order="F")
        z_diag = jnp.array(z_diag, dtype=jnp.complex128)
        try:
            return UnitaryMatrixJax(jnp.diag(z_diag))
        except:
            print(params)
            return UnitaryMatrixJax(jnp.diag(z_diag))

    def optimize(self, env_matrix: jpt.NDArray[jnp.complex128], get_untry=False, prev_utry: UnitaryMatrixJax = None, beta: float = 0.0) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """

        # Ignoring beta and prev_utry right now

        diag = jnp.diag(env_matrix)

        a = jnp.angle(diag[0::2])
        b = jnp.angle(diag[1::2])
        new_thetas = b - a

        if get_untry:
            return self.get_unitary(new_thetas)
    
        return new_thetas
    
    @property
    def name(self) -> str:
        """The name of this gate."""
        base_name = getattr(self, '_name', self.__class__.__name__)
        return f"{base_name}_{self.num_qudits}"
    
    @staticmethod
    def get_params(utry: jpt.NDArray[jnp.complex128]) -> RealVector:
        """Return the params for this gate, given a unitary matrix."""
        diag = jnp.diag(utry)
        a = jnp.angle(diag[0::2]) * 2
        return list(np.array(a, dtype=np.float64))