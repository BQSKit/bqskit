"""This module implements the ControlledGate class."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.gates.quditgate import QuditGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.typing import is_integer
from itertools import product
from functools import reduce
from bqskit.qis.unitary import IntegerVector


class ControlledGate(ComposedGate,QuditGate,DifferentiableUnitary):
    """
    An arbitrary controlled gate.

    Given any qudit gate, ControlledGate can add control a control qudit.
    """

    def __init__(self, gate: Gate, num_levels: int=2, controls: IntegerVector=[1]):
        """
        Construct a ControlledGate.

        Args:
            gate (Gate): The gate to control.
            
            num_levels (int): the number of levels in the qudit

            controls list(int): The levels of control qudits.

        Raises:
            ValueError: If `num_levels` is less than 2 or not a positive integer
                        If level >= num_levels
                        IF Gate.radixes != num_levels
        """

        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s.' % type(gate))

        if num_levels <2 or not is_integer(num_levels):
            raise ValueError(
                'num_levels must be a postive integer greater than or equal to 2.',
            )

        for level in controls:
            if level >= num_levels:
                raise ValueError(
                    'level must be less than the radixes',
                )
        if gate.num_levels != num_levels:
            raise ValueError('The radixes of the Gate must be equal to the number of levels',)

        self.gate = gate
        self.controls = controls
        self._num_controls = len(controls)
        self.num_levels = num_levels
        self._num_qudits = gate._num_qudits + self._num_controls
        #self.radixes = tuple([self.num_levels] * self._num_qudits)
        self.control_space = self._get_control_levels()
        self._name = 'Controlled(%s)' % gate.name
        self._num_params = gate._num_params
        self.It = np.identity(gate.dim, dtype=np.complex128)
        
        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            U = self.gate.get_unitary()
            self.utry = UnitaryMatrix(self._unitary(U), self.radixes)

    def _get_control_levels(self):
        control_levels = lambda x: list(product(*x))
        copies=[range(self.num_levels) for i in range(self._num_controls)]
        return control_levels(copies)
        
    
    def _unitary(self,U):
        M = np.zeros((self.num_levels**self._num_qudits, self.num_levels**self._num_qudits),dtype=complex)
        for el in self.control_space:
            vecs = [np.zeros(self.num_levels) for i in range(self._num_controls)]
            for i in range(len(el)):
                vecs[i][el[i]]=1.0
            ops = [np.outer(v,v) for v in vecs]
            control = reduce(np.kron,ops)
            if el == tuple(self.controls):
                M+=np.kron(control, U)
            else:
                M+=np.kron(control, self.It)
        return M
    
    def _grad(self,grads):
        el = tuple(self.controls)
        vecs = [np.zeros(self.num_levels) for i in range(self._num_controls)]
        for i in range(len(el)):
            vecs[i][el[i]]=1.0
        ops = [np.outer(v,v) for v in vecs]
        control = reduce(np.kron,ops)
        
        result=[]
        for i in range(len(grads)):
            result.append(np.kron(control, grads[i]))
        return np.array(result,dtype=np.complex128)
        
    
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, 'utry'):
            return self.utry

        U = self.gate.get_unitary(params)
        return UnitaryMatrix(self._unitary(U))

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry') or self._num_params==0:
            return np.array([])

        grads = self.gate.get_grad(params)  # type: ignore
        return self._grad(grads)


    def get_unitary_and_grad(self, params: RealVector = [],) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return self.utry, np.array([])
        elif self._num_params==0:
            return np.array([]), np.array([])

        U, G = self.gate.get_unitary_and_grad(params)  # type: ignore

        utry = UnitaryMatrix(self._unitary(U))
        grads = self._grad(G)
        return utry, grads


    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ControlledGate)
            and self.gate == other.gate
            and self.controls == other.controls
            and self.num_levels == other.num_levels
        )

    def __hash__(self) -> int:
        return hash((self.gate, self.radixes))