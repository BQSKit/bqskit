"""This module implements the RXGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import jax
jax.config.update("jax_enable_x64", True)

from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.ir.gates.qutrit.constant.x import X01Gate, X02Gate, X12Gate
        
class RX01Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the X axis for qutrits

    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rx01'

    x = np.array(X01Gate._utry)
    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*self.x)
    
    
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
        
class RX02Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the X axis for qutrits

    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rx02'

    x = np.array(X02Gate._utry)
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*self.x)
    
    
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
        
class RX12Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the X axis for qutrits

    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rx12'

    x = np.array(X12Gate._utry)
    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*self.x)
    
    
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
