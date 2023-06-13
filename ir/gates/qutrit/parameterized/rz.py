"""This module implements the RZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import jax

from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.ir.gates.qutrit.constant.z import ZGate, Z0Gate, Z1Gate, Z2Gate


class RZGate(QutritGate, DifferentiableUnitary, CachedClass):
    """
    A gate representing an arbitrary rotation around the Z axis for qutrits

    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz'

    z = ZGate._utry
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*z)
    
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
        
class RZ0Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """
    A gate representing an arbitrary rotation around the Z axis for qutrits

    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz0'

    z = ZGate._utry
    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*z)
    
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

class RZ1Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """
    A gate representing an arbitrary rotation around the Z axis for qutrits

    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz1'

    z = Z1Gate._utry
    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*z)
    
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
        
        
class RZ2Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """
    A gate representing an arbitrary rotation around the Z axis for qutrits

    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz2'

    z = Z2Gate._utry
    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*z)
    
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