"""This module implements the RXXGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import jax

from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.ir.gates.qutrit.constant.xx import XXGate, X01X01Gate, X02X02Gate, X01X02Gate, X02X01Gate


class RXXGate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the XX axis for qutrits

    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rxx'

    xx = XXGate._utry
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*xx)
    
    
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return UnitaryMatrix(self._unitary(params).tolist())
            
       
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
        
        
class RX01X01Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the XX axis for qutrits

    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rx01x01'
    
    xx = X01X01Gate._utry
    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*xx)
    
    
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return UnitaryMatrix(self._unitary(params).tolist())
            
       
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
    """
    A gate representing an arbitrary rotation around the XX axis for qutrits

    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rx02x02'
    
    xx = X02X02Gate._utry

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*xx)
    
    
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return UnitaryMatrix(self._unitary(params).tolist())
            
       
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
    """
    A gate representing an arbitrary rotation around the XX axis for qutrits

    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rx01x02'
    
    xx = X01X02Gate._utry

    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*xx)
    
    
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return UnitaryMatrix(self._unitary(params).tolist())
            
       
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
    """
    A gate representing an arbitrary rotation around the XX axis for qutrits

    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rx02x01'

    xx = X02X01Gate._utry
    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*xx)
    
    
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)
        return UnitaryMatrix(self._unitary(params).tolist())
            
       
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
    
