"""This module implements the ControlledGate class."""
from __future__ import annotations

from functools import reduce
from itertools import product

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.gates.quditgate import QuditGate
from typing import Sequence
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.typing import is_integer


class ControlledGate(ComposedGate, QuditGate, DifferentiableUnitary):
    """
    An arbitrary controlled gate.
    Given any qudit gate, ControlledGate can add control a control qudit.
    """
    def __init__(self, gate: Gate, num_controls: int=1, num_levels: Sequence[int] = [2] | int = 2, level_of_each_control: Sequence[Sequence[int]] = [[1]]):
        """
        Construct a ControlledGate.

        Args:
            gate (Gate): The gate to control.

            num_controls (int): the number of controls 
            
            num_levels (Sequence[int]): the number of levels for each control qudit. 
            If one number is provided, all qudits have the same number of levels

            level_of_each_control  (Sequence[Sequence[int]]): Sequence of control levels for each control qudit.
            If more than one level is selected, the subspace spanned by the levels acts as a control subspace.
            If all levels are selected for a given qudit, the operation is equivalent to the original gate without controls.  


        Raises:
            TypeError: If gate is not of type Gate

            ValueError: If `num_controls` is less than 1 or not a positive integer
                        If `num_levels` is less than 2 or not a positive integer
                        If len(num_levels) != num_controls
                        If len(level_of_each_control) != num_controls 
                        If np.any(level_of_each_control[i] >= num_levels[i])
        """

        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s.' % type(gate))

        if num_controls < 1 or not is_integer(num_controls):
            raise ValueError(
                'num_controls must be a postive integer greater than or equal to 1.',
            )
        if type(num_levels) == Sequence[int] or num_levels < 2 or not is_integer(num_levels):
            raise ValueError(
                'num_levels must be a postive integer >= 2 or sequence of such integers.',
            )
        if type(num_levels) == Sequence[int]:
            if len(num_levels)!=num_controls:
                raise ValueError(
                    'Sequence of number of levels must have the same length as the number of controls.',
                )   
        if len(level_of_each_control)!= num_controls:
            raise ValueError(
                'Sequence of levels of each control must have the same length as the number of controls.',
            )
        for level in controls:
            if level >= num_levels:
                raise ValueError(
                    'level must be less than the radixes',
                )
        
        self.gate = gate
        self._num_controls = num_controls
        
        if type(num_levels)==int:
            self.num_levels = [num_levels for i in range(len(self._num_controls))]
        elif type(num_levels) == Sequence[int]:
            self.num_levels = num_levels

        for i in range(len(level_of_each_control)):
            if np.any(level_of_each_control[i]>=num_levels[i]):
                raise ValueError(
                    'Levels of control qubit must be less than the number of levels.',
                )

        self.level_of_each_control = level_of_each_control
        self._num_qudits = gate._num_qudits + self._num_controls
        # self.radixes = tuple([self.num_levels] * self._num_qudits)
        self.control_space = self._get_control_levels()
        self._name = 'Controlled(%s)' % self.gate.name
        self._num_params = self.gate._num_params
        self.It = np.identity(self.gate.dim, dtype=np.complex128)

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            U = self.gate.get_unitary()
            self.utry = UnitaryMatrix(self._unitary(U), self.radixes)

   def _get_control_matrix(self):
        Matrix_list = [np.zeros((self.num_levels[i],self.num_levels[i]),dtype=np.complex128) for i in range(self._num_controls)]
        for i in range(self._num_controls):
            for j in self.level_of_each_control[i]:
                Matrix_list[i][j,j]=1.0
        result = reduce(np.kron,Matrix_list)
        return result

    def _unitary(self, U):
        dim = int(np.prod(self.num_levels))
        M_other = np.eye(dim, dtype=np.complex128)
        M_control = self._get_control_matrix()
        M_other -= M_control
        return np.kron(M_control, U) + np.kron(M_other,self.It)
        
    def _grad(self, grads):
        M_control = self._get_control_matrix()

        result = []
        for i in range(len(grads)):
            result.append(np.kron(M_control, grads[i]))
        return np.array(result, dtype=np.complex128)

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
        if hasattr(self, 'utry') or self._num_params == 0:
            return np.array([])

        grads = self.gate.get_grad(params)  # type: ignore
        return self._grad(grads)

    def get_unitary_and_grad(self, params: RealVector = []) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return self.utry, np.array([])
        elif self._num_params == 0:
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
