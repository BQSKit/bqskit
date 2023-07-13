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


class R1Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """The R1 rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'R1Gate'

    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*Lambda[1])
    
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
    
class R2Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """The R2 rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'R2Gate'

    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*Lambda[2])
    
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
    
class R3Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """The R3 rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'R3Gate'

    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*Lambda[3])
    
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
    
class R4Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """The R4 rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'R4Gate'

    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*Lambda[4])
    
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
    
class R5Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """The R5 rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'R5Gate'

    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*Lambda[5])
    
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

class R6Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """The R6 rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'R6Gate'

    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*Lambda[6])
    
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
    
class R7Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """The R7 rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'R7Gate'

    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*Lambda[7])
    
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
    
class R8Gate(QutritGate, DifferentiableUnitary, CachedClass):
    """The R8 rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'R8Gate'

    
    def _unitary(self, params: RealVector = []):
        return jax.scipy.linalg.expm(-1j*params[0]*Lambda[8])
    
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
    
    
class RGVGate(QutritGate, DifferentiableUnitary, CachedClass):
    """The Rvec rotation single qutrit gate."""

    _num_qudits = 1
    _num_params = 8
    _qasm_name = 'RGVGate'

    
    def _unitary(self, params: RealVector = []):
        tot = jax.numpy.sum(jax.numpy.array([params[i]*Lambda[i+1] for i in range(8)]),axis=0)
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