"""This module implements the CRXGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.ir.gates.qutrit.constant.x import XGate, X01Gate, X02Gate, X12Gate
from bqskit.ir.gates.qutrit import jaxcgate

class CRXGate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a controlled X rotation for qutrits

    
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crx'

    x = XGate._utry
    
    def _unitary(self, params):
        return jaxcgate(params[0],jax.scipy.linalg.expm(-1j*params[1]*x))
    
    def _unitary_diff(self, params):
        return jax.numpy.kron(pdict[params[0]],jax.scipy.linalg.expm(-1j*params[1]*x))
    
    def get_unitary(self, params):
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self._unitary(params).tolist()
            
       
    def get_grad(self, params) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        def _temp(param):
            return self._unitary_diff([params[0],param])
      
        return np.array(
            [ np.zeros((9,9)),
                np.array(jax.jacfwd(_temp)(params[1])),
            ], dtype=np.complex128,
        )
    
class CRX01Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a controlled X rotation for qutrits

    
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crx01'

    x = X01Gate._utry
    
    def _unitary(self, params):
        return jaxcgate(params[0],jax.scipy.linalg.expm(-1j*params[1]*x))
    
    def _unitary_diff(self, params):
        return jax.numpy.kron(pdict[params[0]],jax.scipy.linalg.expm(-1j*params[1]*x))
    
    def get_unitary(self, params):
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self._unitary(params).tolist()
            
       
    def get_grad(self, params) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        def _temp(param):
            return self._unitary_diff([params[0],param])
      
        return np.array(
            [ np.zeros((9,9)),
                np.array(jax.jacfwd(_temp)(params[1])),
            ], dtype=np.complex128,
        )
    
class CRX02Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a controlled X rotation for qutrits

    
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crx02'

    x = X02Gate._utry
    
    def _unitary(self, params):
        return jaxcgate(params[0],jax.scipy.linalg.expm(-1j*params[1]*x))
    
    def _unitary_diff(self, params):
        return jax.numpy.kron(pdict[params[0]],jax.scipy.linalg.expm(-1j*params[1]*x))
    
    def get_unitary(self, params):
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self._unitary(params).tolist()
            
       
    def get_grad(self, params) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        def _temp(param):
            return self._unitary_diff([params[0],param])
      
        return np.array(
            [ np.zeros((9,9)),
                np.array(jax.jacfwd(_temp)(params[1])),
            ], dtype=np.complex128,
        )
    
class CRX12Gate(
    QutritGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing a controlled X rotation for qutrits

    
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'crx12'

    x = X12Gate._utry
    
    def _unitary(self, params):
        return jaxcgate(params[0],jax.scipy.linalg.expm(-1j*params[1]*x))
    
    def _unitary_diff(self, params):
        return jax.numpy.kron(pdict[params[0]],jax.scipy.linalg.expm(-1j*params[1]*x))
    
    def get_unitary(self, params):
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        return self._unitary(params).tolist()
            
       
    def get_grad(self, params) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        def _temp(param):
            return self._unitary_diff([params[0],param])
      
        return np.array(
            [ np.zeros((9,9)),
                np.array(jax.jacfwd(_temp)(params[1])),
            ], dtype=np.complex128,
        )