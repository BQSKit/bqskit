"""This module implements the ControlledGate class."""
from __future__ import annotations

from functools import reduce
from typing import Sequence

import numpy as np
import numpy.typing as npt

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.docs import building_docs
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence
from bqskit.utils.typing import is_sequence_of_int


class ControlledGate(ComposedGate, DifferentiableUnitary):
    """
    An arbitrary controlled gate.

    Given any qudit gate, ControlledGate can add control qudits.

    A controlled gate adds arbitrarily controls, and can be generalized
    for qudit or even mixed-qudit representation.

    A controlled gate has a circuit structure as follows:

    ..
        controls ----/----■----
                          |
                         .-.
        targets  ----/---|G|---
                         '-'

    Where `G` is the gate being controlled.

    To calculate the unitary for a controlled gate, given the unitary of
    the gate being controlled, we can use the following equation:

    .. math::
        U_{control} = P_i \\otimes I + P_c \\otimes G

    Where :math:`P_i` is the projection matrix for the states that don't
    activate the gate, :math:`P_c` is the projection matrix for the
    states that do activate the gate, :math:`I` is the identity matrix
    of dimension equal to the gate being controlled, and :math:`G` is
    the unitary matrix of the gate being controlled.

    In the simple case of a normal qubit CNOT, :math:`P_i` and :math:`P_c`
    are defined as follows:

    .. math::

        P_i = |0\\rangle\\langle 0|
        P_c = |1\\rangle\\langle 1|

    This is because the :math:`|0\\rangle` state is the state that doesn't
    activate the gate, and the :math:`|1\\rangle` state is the state that
    does activate the gate.

    We can also decide to invert this, and have the :math:`|0\\rangle`
    state activate the gate, and the :math:`|1\\rangle` state not activate
    the gate. This is equivalent to swapping :math:`P_i` and :math:`P_c`,
    and usually drawn diagrammatically as follows:

    ..
        controls ----/----□----
                          |
                         .-.
        targets  ----/---|G|---
                         '-'


    When we add more controls the projection matrices become more complex,
    but the basic idea stays the same: we have a projection matrix for
    the states that activate the gate, and a projection matrix for the
    states that don't activate the gate. As in the case of a toffoli gate,
    the projection matrices are defined as follows:

    .. math::

        P_i = |00\\rangle\\langle 00| + |01\\rangle\\langle 01|
            + |10\\rangle\\langle 10|

        P_c = |11\\rangle\\langle 11|

    This is because the :math:`|00\\rangle`, :math:`|01\\rangle`, and
    :math:`|10\\rangle` states are the states that don't activate the
    gate, and the :math:`|11\\rangle` state is the state that does
    activate the gate.

    With qudits, we have more states and as such, more complex
    projection matrices; however, the basic idea is the same.
    For example, a qutrit controlled-not gate that is activated by
    the :math:`|2\\rangle` state and not activated by the :math:`|0\\rangle`
    and :math:`|1\\rangle` states is defined as follows:

    .. math::

        P_i = |0\\rangle\\langle 0| + |1\\rangle\\langle 1|
        P_c = |2\\rangle\\langle 2|

    One interesting concept with qudits is that we can have multiple
    active control levels. For example, a qutrit controlled-not gate that
    is activated by the :math:`|1\\rangle` and :math:`|2\\rangle` states
    and not activated by the :math:`|0\\rangle` state is defined similarly
    as follows:

    .. math::

        P_i = |0\\rangle\\langle 0|
        P_c = |1\\rangle\\langle 1| + |2\\rangle\\langle 2|

    Note that we can always define :math:`P_i` simply from :math:`P_c`:

    .. math::

        P_i = I_p - P_c

    Where :math:`I_p` is the identity matrix of dimension equal to the
    dimension of the control qudits. This leaves us with out final
    equation:

    .. math::

        U_{control} = (I_p - P_c) \\otimes I + P_c \\otimes G

    If, G is a unitary-valued function of real parameters, then the
    gradient of the controlled gate simply discards the constant half
    of the equation:

    .. math::

        \\frac{\\partial U_{control}}{\\partial \\theta} =
            P_c \\otimes \\frac{\\partial G}{\\partial \\theta}
    """

    def __init__(
        self,
        gate: Gate,
        num_controls: int = 1,
        control_radixes: Sequence[int] | int = 2,
        control_levels: Sequence[Sequence[int] | int] | int | None = None,
    ):
        """
        Construct a ControlledGate.

        Args:
            gate (Gate): The gate to control.

            num_controls (int): The number of controls.

            control_radixes (Sequence[int] | int): The number of levels
                for each control qudit. If one number is provided, all
                control qudits will have the same number of levels.
                Defaults to qubits (2).

            control_levels  (Sequence[Sequence[int] | int] | int | None):
                Sequence of control levels for each control qudit. These
                levels need to be activated on the corresponding control
                qudits for the gate to be activated. If more than one
                level is selected, the subspace spanned by the levels
                acts as a control subspace. If all levels are selected
                for a given qudit, the operation is equivalent to the
                original gate without controls. If None, the highest
                level acts as the control level for each control qudit.
                Can be given as a single integer, which will be broadcast
                as the control level for all control qudits; or as a
                sequence, where each element describes the control levels
                for the corresponding control qudit. These can be given
                as a single integer or a sequence of integers. Defaults
                to None.

        Raises:
            ValueError: If `num_controls` is less than 1

            ValueError: If any control radix is less than 2

            ValueError: If any control level is less than 0

            ValueError: If the length of control_radixes is not equal to
                num_controls

            ValueError: If the length of control_levels is not equal to
                num_controls

            ValueError: If any control level is repeated in for the same
                qudit

            ValueError: If there exists an `i` and `j` where
                `control_levels[i][j] >= control_radixes[i]`

        Examples:
            If we didn't have the CNOTGate we can do it from this gate:

            >>> from bqskit.ir.gates import XGate, ControlledGate
            >>> cnot_gate = ControlledGate(XGate())
            >>> cnot_gate.get_unitary()
            array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                   [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                   [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                   [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])

            We can invert the controls of a CNOTGate, by activating on the
            |0> state instead of the |1> state:

            >>> from bqskit.ir.gates import XGate, ControlledGate
            >>> inverted_cnot_gate = ControlledGate(XGate(), control_levels=0)

            Also, if we didn't have the ToffoliGate for qubits, we could do:

            >>> from bqskit.ir.gates import XGate, ControlledGate, ToffoliGate
            >>> toffoli_gate = ControlledGate(XGate(), 2) # 2 controls
            >>> ToffoliGate().get_unitary() == toffoli_gate.get_unitary()
            True

            We can define a qutrit CNOT that is activated by the |2> state:

            >>> from bqskit.ir.gates import ShiftGate, ControlledGate
            >>> qutrit_x = ShiftGate(3)
            >>> qutrit_cnot = ControlledGate(qutrit_x, control_radixes=3)

            This composed gate can also be used to define mixed-radix
            controlled systems. For example, we can define a mixed-radix
            CNOT where the control is a qubit and the X Gate is qutrit:

            >>> from bqskit.ir.gates import ShiftGate, ControlledGate
            >>> qutrit_x = ShiftGate(3)
            >>> qutrit_cnot = ControlledGate(qutrit_x)

            We can also define multiple controls with mixed qudits
            that require multiple levels to be activated. In this
            example, The first control is a qutrit with [0,1] control
            levels, the second qudit is a ququart with a [0] control level,
            and RY Gate for qubit operation:

            >>> from bqskit.ir.gates import RYGate, ControlledGate
            >>> cgate = ControlledGate(
            ...     RYGate(),
            ...     num_controls=2,
            ...     control_radixes=[3,4],
            ...     control_levels=[[0,1], 0]
            ... )
        """
        if not isinstance(gate, Gate):
            raise TypeError(f'Expected gate object, got {type(gate)}.')

        self.gate = gate

        params = self._check_and_type_control_parameters(
            num_controls,
            control_radixes,
            control_levels,
        )
        num_controls, control_radixes, control_levels = params
        self.num_controls, self.control_radixes, self.control_levels = params

        self._radixes = tuple(tuple(control_radixes) + self.gate.radixes)
        self._num_qudits = gate._num_qudits + self.num_controls
        # TODO: Incorporate control radixes/levels into name with function def.
        self._name = 'Controlled(%s)' % self.gate.name
        self._num_params = self.gate._num_params

        iden_gate = np.identity(self.gate.dim, dtype=np.complex128)
        """Identity is applied when controls are not properly activated."""

        self.ctrl = self.build_control_proj(control_radixes, control_levels)
        """Control projection matrix determines if the gate should activate."""

        ctrl_dim = int(np.prod(self.control_radixes))
        """Dimension of the control qudits."""

        iden_proj = np.eye(ctrl_dim, dtype=np.complex128) - self.ctrl
        """Identity projection matrix determines if it shouldn't activate."""

        self.ihalf = np.kron(iden_proj, iden_gate)
        """Identity half of the final unitary equation."""

        # If input is a constant gate, we can cache the unitary.
        if self.num_params == 0 and not building_docs():
            U = self.gate.get_unitary()
            ctrl_U = np.kron(self.ctrl, U) + self.ihalf
            self._utry = UnitaryMatrix(ctrl_U, self.radixes)

    @property
    def qasm_name(self) -> str:
        """
        Override default `Gate.qasm_name` method.

        If the core gate is a standard gate, this function will output
        qasm in the form 'c+<gate_qasm>'. Otherwise an error will be raised.

        Raises:
            ValueError: If the core gate is non-standard in OpenQASM 2.0.
        """
        _core_gate = self.gate.qasm_name
        if self.num_controls <= 2:
            _controls = 'c' * self.num_controls
        else:
            _controls = f'c{self.num_controls}'
        qasm_name = _controls + _core_gate
        supported_gates = ('cu1', 'cu2', 'cu3', 'cswap', 'c3x', 'c4x')
        if qasm_name not in supported_gates:
            raise ValueError(
                f'Controlled gate {_core_gate} with {self.num_controls} '
                'controls is not a standard OpenQASM 2.0 identifier. '
                'To encode this gate, try decomposing it into gates with'
                'standard identifiers.',
            )
        return qasm_name

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if hasattr(self, '_utry'):
            return self._utry

        U = self.gate.get_unitary(params)
        ctrl_U = np.kron(self.ctrl, U) + self.ihalf
        return UnitaryMatrix(ctrl_U, self.radixes)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, '_utry'):
            return np.array([])

        grads = self.gate.get_grad(params)  # type: ignore
        return np.kron(self.ctrl, grads)

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return the unitary and gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if hasattr(self, '_utry'):
            return self._utry, np.array([])

        U, grads = self.gate.get_unitary_and_grad(params)  # type: ignore
        ctrl_U = np.kron(self.ctrl, U) + self.ihalf
        ctl_grads = np.kron(self.ctrl, grads)
        return UnitaryMatrix(ctrl_U, self.radixes), ctl_grads

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ControlledGate)
            and self.gate == other.gate
            and self.num_controls == other.num_controls
            and self.control_radixes == other.control_radixes
            and self.control_levels == other.control_levels
        )

    def __hash__(self) -> int:
        return hash((self.gate, self.radixes))

    @staticmethod
    def build_control_proj(
        control_radixes: Sequence[int],
        control_levels: Sequence[Sequence[int]],
    ) -> npt.NDArray[np.complex128]:
        """
        Construct the control projection matrix from the control levels.

        See :class:`ControlledGate` for more info.
        """
        params = ControlledGate._check_and_type_control_parameters(
            len(control_radixes),
            control_radixes,
            control_levels,
        )
        _, control_radixes, control_levels = params

        elementary_projection_list = [
            np.zeros((r, r), dtype=np.complex128)
            for r in control_radixes
        ]

        for i, projection in enumerate(elementary_projection_list):
            for control_level in control_levels[i]:
                projection[control_level, control_level] = 1.0

        return reduce(np.kron, elementary_projection_list)  # type: ignore

    @staticmethod
    def _check_and_type_control_parameters(
        num_controls: int = 1,
        control_radixes: Sequence[int] | int = 2,
        control_levels: Sequence[Sequence[int] | int] | int | None = None,
    ) -> tuple[int, list[int], list[list[int]]]:
        """
        Checks the control paramters for type and value errors.

        Returns specific types for each parameter; see
        :class:`ControlledGate` for more info on errors and parameters.
        """
        if not is_integer(num_controls):
            raise TypeError(
                f'Expected integer for num_controls, got {type(num_controls)}.',
            )

        if num_controls < 1:
            raise ValueError(
                'Expected num_controls to be greater than or equal to 1'
                f', got {num_controls}.',
            )

        if is_integer(control_radixes):
            control_radixes = [control_radixes] * num_controls

        if not is_sequence_of_int(control_radixes):
            raise TypeError(
                'Expected integer or sequence of integers for control_radixes.',
            )

        if any(r < 2 for r in control_radixes):
            raise ValueError('Expected every radix to be greater than 1.')

        if len(control_radixes) != num_controls:
            raise ValueError(
                'Expected length of control_radixes to be equal to '
                f'num_controls: {len(control_radixes)=} != {num_controls=}.',
            )

        if control_levels is None:
            control_levels = [
                [control_radixes[i] - 1]
                for i in range(num_controls)
            ]

        if is_integer(control_levels):
            control_levels = [[control_levels]] * num_controls

        if not is_sequence(control_levels):
            raise TypeError(
                'Expected sequence of sequence of integers for control_levels,'
                f'got {type(control_levels)}.',
            )

        control_levels = [
            [level] if is_integer(level) else level
            for level in control_levels
        ]

        if any(not is_sequence_of_int(levels) for levels in control_levels):
            bad = [not is_sequence_of_int(levels) for levels in control_levels]
            bad_index = bad.index(True)
            raise TypeError(
                'Expected sequence of sequence of integers for control_levels,'
                f'got {control_levels[bad_index]} where a sequence of integers'
                ' was expected.',
            )

        if len(control_levels) != num_controls:
            raise ValueError(
                'Expected length of control_levels to be equal to '
                f'num_controls: {len(control_levels)=} != {num_controls=}.',
            )

        if any(l < 0 for levels in control_levels for l in levels):
            raise ValueError(
                'Expected control levels to be greater than or equal to 0.',
            )

        if any(
            l >= r
            for r, levels in zip(control_radixes, control_levels)
            for l in levels
        ):
            raise ValueError(
                'Expected control levels to be less than the number of levels.',
            )

        if any(len(l) != len(set(l)) for l in control_levels):
            raise ValueError(
                'Expected control levels to be unique for each qudit.',
            )

        control_levels = [list(levels) for levels in control_levels]
        return num_controls, list(control_radixes), control_levels
