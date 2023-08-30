"""This module implements the ControlledGate class."""
from __future__ import annotations

from functools import reduce
from typing import Sequence

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


class ControlledGate(ComposedGate, QuditGate, DifferentiableUnitary):
    """
    An arbitrary controlled gate.

    Given any qudit gate, ControlledGate can add control qudits.
    """

    def __init__(
        self,
        gate: Gate,
        num_controls: int = 1,
        num_control_levels: Sequence[int] | int = 2,
        level_of_each_control: Sequence[Sequence[int]] | None = None,
    ):
        """
        Construct a ControlledGate.

        A controlled gate adds arbitrarily controls,
        and can be generalized for mixed qudit representation.

        Args:
            gate (Gate): The gate to control.

            num_controls (int): number of controls

            num_control_levels (Sequence[int]): number of levels for each control qudit.
            If one number is provided, all qudits have the same number of levels

            level_of_each_control  (Sequence[Sequence[int]]): Sequence of control levels for each control qudit.
            If more than one level is selected, the subspace spanned by the levels acts as a control subspace.
            If all levels are selected for a given qudit, the operation is equivalent to the original gate without controls.


        Raises:
            TypeError: If gate is not of type Gate
            TypeError: If num_controls is not an integer
            TypeError: If num_control_levels is not an integer or sequence of integers
            TypeError: If level_of_each_control is not a sequence of sequences of integers

            ValueError: If `num_controls` is less than 1
                        If `num_control_levels` is less than 2
                        If len(num_control_levels) != num_controls
                        If len(level_of_each_control) != num_controls
                        If np.any(level_of_each_control[i] >= num_control_levels[i])

        Examples:
            CNOT for qubits:
                If we didn't have the CNOTGate we can do it from this gate:
                ```
                > cnot_gate = ControlledGate(XGate())
                > cnot_gate.get_unitary()
                > [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]]
                ```

            Toffoli for qubits:
                ```
                >toffoli_gate = ControlledGate(XGate())
                >toffoli_gate.get_unitary()
                ```

            CNOT for qutrits:
                ```
                >cnot_gate = ControlledGate(XGate(num_levels=3),num_levels=3)
                >cnot_gate.get_unitary()
                ```

            Hybrid CNOT: control is qubit, X Gate is qutrit:
                ```
                >cnot_gate = ControlledGate(XGate(num_levels=3))
                >cnot_gate.get_unitary()
                ```

            Multiple controls with mixed qudits: first control is qutrit with [0,1] control levels,
            second qudit is a 4 level qubit with [0] control,
            and RY Gate for qubit operation:
               ```
                >cgate = ControlledGate(RYGate(),num_controls=2,num_levels=[3,4],level_of_each_control=[[0,1],[0]])
                >cgate.get_unitary(params=[0.3])
                ```
        """
        if not isinstance(gate, Gate):
            raise TypeError('Expected gate object, got %s.' % type(gate))

        if not is_integer(num_controls):
            raise TypeError('num_controls must be an integer')

        if not is_integer(num_control_levels) and type(num_control_levels) != Sequence[int]:
            raise TypeError(
                'num_control_levels must be an integer or sequence of integers',
            )

        if num_controls < 1:
            raise ValueError(
                'num_controls must be a postive integer greater than or equal to 1.',
            )
        if np.any(np.array(num_control_levels) < 2):
            raise ValueError(
                'num_levels must be a postive integer >= 2 or sequence of such integers.',
            )
        if type(num_control_levels) == Sequence[int]:
            if len(num_control_levels) != num_controls:
                raise ValueError(
                    'Sequence of number of levels must have the same length as the number of controls.',
                )
        if type(num_control_levels) == int:
            _num_control_levels = [
                int(num_control_levels)
                for i in range(num_controls)
            ]
        elif type(num_control_levels) == Sequence[int]:
            _num_control_levels = [level for level in num_control_levels]

        if level_of_each_control is None:
            _level_of_each_control = []
            for i in range(num_controls):
                _level_of_each_control.append([_num_control_levels[i] - 1])
        elif type(level_of_each_control) != Sequence[Sequence[int]]:
            raise TypeError(
                'level_of_each_control type must be Seqence[Sequnce[int]] but is %s' %type(level_of_each_control),
            )

        if type(level_of_each_control) == Sequence[Sequence[int]]:
            if len(level_of_each_control) != num_controls:
                raise ValueError(
                    'Sequence of levels of each control must have the same length as the number of controls.',
                )
            for i in range(len(level_of_each_control)):
                if np.any(np.array(level_of_each_control[i]) >= num_control_levels[i]):
                    raise ValueError(
                        'Levels of control qubit must be less than the number of levels.',
                    )
            _level_of_each_control = []
            for level in level_of_each_control:
                _level_of_each_control.append(list(level))

        self.gate = gate
        self._num_controls = num_controls
        self._num_levels = _num_control_levels+list(self.gate.radixes)
        self._radixes= tuple(self._num_levels)
        self._num_control_levels = _num_control_levels
        self._level_of_each_control = _level_of_each_control
        self._num_qudits = gate._num_qudits + self._num_controls
        self._name = 'Controlled(%s)' % self.gate.name
        self._num_params = self.gate._num_params
        self.It = np.identity(self.gate.dim, dtype=np.complex128)

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            U = self.gate.get_unitary()
            self.utry = UnitaryMatrix(self._unitary(U), self.radixes)

    def _get_control_matrix(self) -> npt.NDArray[np.complex128]:
        """
        Construct the control matrix from the level of the control qudits.

        At first, for each control qudit, it creates a square matrix of zeros of
        dimensionality equal to the number of levels. Then for each control
        level, the corresponding diagonal element is set to 1. This is
        equivalent to adding the respective state projection matrix. The final
        result matrix is the kronecker product of the indvidual matrices.
        """

        Matrix_list = [
            np.zeros(
                (self._num_control_levels[i], self._num_control_levels[i]),
                dtype=np.complex128,
            ) for i in range(self._num_controls)
        ]
        for i in range(self._num_controls):
            for j in self._level_of_each_control[i]:
                Matrix_list[i][j, j] = 1.0
        result = reduce(np.kron, Matrix_list)
        return result

    def _unitary(self, U: UnitaryMatrix) -> npt.NDArray[np.complex128]:
        """
        Returns the unitary for the cotrol operation based on gate with unitary
        matrix U.

        The control matrix is obtained from self._get_control_matrix(), and the
        complentary matrix is obtained from subtrating this matrix from the
        Idenity. Then, each matrix is Kronecker multiplied with U and Idenity on
        active qudits respectively. The operation is equal to the sum of these
        two operators.
        """
        dim = int(np.prod(self._num_control_levels))
        M_other = np.eye(dim, dtype=np.complex128)
        M_control = self._get_control_matrix()
        M_other -= M_control
        return np.kron(M_control, U) + np.kron(M_other, self.It)

    def _grad(self, grads: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Returns the gradient of the controlled unitary based on the gradient of
        the original gate.

        The control matrix is obtained from self._get_control_matrix(), and
        Kronecker multiplied with each gradient element.
        """

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
        elif self.num_params > 0 and hasattr(self.gate, 'get_grad'):
            grads = self.gate.get_grad(params)
            return self._grad(grads)
        else:
            return np.array([])

    def get_unitary_and_grad(self, params: RealVector = []) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, 'utry'):
            return self.utry, np.array([])
        elif self._num_params == 0:
            return UnitaryMatrix(np.array([])), np.array([])
        elif self.gate._num_params > 0 and hasattr(self.gate, 'get_unitary_and_grad'):
            U, G = self.gate.get_unitary_and_grad(params)

            utry = UnitaryMatrix(self._unitary(U))
            grads = self._grad(G)
            return utry, grads
        else:
            return UnitaryMatrix(np.array([])), np.array([])

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ControlledGate)
            and self.gate == other.gate
            and self._num_controls == other._num_controls
            and self._num_control_levels == other._num_control_levels
            and self._level_of_each_control == other._level_of_each_control
        )

    def __hash__(self) -> int:
        return hash((self.gate, self.radixes))
