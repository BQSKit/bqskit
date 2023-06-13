"""This module implements the rotations due to the generatos of SU(3)."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import jax
jax.config.update("jax_enable_x64", True)

from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass

Lambda = {
    1 : np.array([[0,1,0],[1,0,0],[0,0,0]]),
    2 : np.array([[0,-1j,0],[1j,0,0],[0,0,0]]),
    3 : np.array([[1,0,0],[0,-1,0],[0,0,0]]),
    4 : np.array([[0,0,1],[0,0,0],[1,0,0]]),
    5 : np.array([[0,0,-1j],[0,0,0],[1j,0,0]]),
    6 : np.array([[0,0,0],[0,0,1],[0,1,0]]),
    7 : np.array([[0,0,0],[0,0,-1j],[0,1j,0]]),
    8 : 1.0/np.sqrt(3)*np.array([[1,0,0],[0,1,0],[0,0,-2]])}


class RGGVGate(QutritGate, DifferentiableUnitary, CachedClass):
    """The G rotation double qutrit gate with the same generator on each qutrit (Gate = exp(-i sum_j param_j dot G_j x G_j) )."""

    _num_qudits = 2
    _num_params = 8
    _qasm_name = 'RGGVGate'

    
    def _unitary(self, params: RealVector = []):
        tot = jax.numpy.sum(jax.numpy.array([params[i]*jax.numpy.kron(Lambda[i+1],Lambda[i+1]) for i in range(8)]),axis=0)
        return jax.scipy.linalg.expm(-1j*tot)
        
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
    
    
class RGGGate(QutritGate, DifferentiableUnitary, CachedClass):
    """The RGG rotation double qutrit gate (exp(-i param sum_j G_j G_j))."""

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'RGGGate'

    
    def _unitary(self, params: RealVector = []):
        tot = jax.numpy.sum(jax.numpy.array([jax.numpy.kron(Lambda[i+1],Lambda[i+1]) for i in range(8)]),axis=0)
        return jax.scipy.linalg.expm(-1j*params[0]*tot)
    
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