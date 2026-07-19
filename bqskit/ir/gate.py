"""
This module implements the Gate base class.

A gate is a potentially-parameterized, immutable, unitary operation that can be
applied to a circuit.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar
from typing import TYPE_CHECKING

from bqskit.ir.location import CircuitLocation
from bqskit.qis.unitary.unitary import Unitary

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from openqudit.expressions import UnitaryExpression
    from bqskit.qis.unitary.unitary import RealVector
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
    from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate


class Gate(Unitary):
    """
    Gate Base Class.

    A `Gate` can optionally wrap an openqudit `UnitaryExpression` via the
    `_expr` attribute, which then powers `name`/`num_params`/`radixes`/`dim`.
    Gates that don't set `_expr` fall back to the old class-level attributes
    (`_num_params`, `_radixes`, ...). Either way, `get_unitary` itself is
    still provided by the gate (e.g. via `ConstantGate`'s cached `_utry`, or
    a hand-written override) rather than by this base class.
    """

    _expr: UnitaryExpression
    _name: str
    _qasm_name: str

    @property
    def name(self) -> str:
        """The name of this gate."""
        if hasattr(self, '_name'):
            return self._name
        if hasattr(self, '_expr'):
            return self._expr.name()
        return self.__class__.__name__

    @property
    def num_params(self) -> int:
        """The number of real parameters this unitary-valued function takes."""
        if hasattr(self, '_expr'):
            return self._expr.num_params()
        return super().num_params

    @property
    def num_qudits(self) -> int:
        """The number of qudits this unitary can act on."""
        if hasattr(self, '_expr'):
            return len(self._expr.radices())
        return super().num_qudits

    @property
    def radixes(self) -> tuple[int, ...]:
        """The number of orthogonal states for each qudit."""
        if hasattr(self, '_expr'):
            return tuple(self._expr.radices())
        return super().radixes

    @property
    def dim(self) -> int:
        """The matrix dimension for this unitary."""
        if hasattr(self, '_expr'):
            return self._expr.dimension()
        return super().dim

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for the unitary map as an np.ndarray.

        Args:
            params (RealVector): The unitary parameters, see
                :func:`Unitary.get_unitary` for more info.

        Returns:
            np.ndarray: The `(num_params,N,N)`-shaped, matrix-by-vector
            derivative of this unitary at the point specified by params.

        Notes:
            The gradient of a unitary is defined as a matrix-by-vector
            derivative. If the UnitaryMatrix result of `get_unitary` has
            dimension NxN, then the shape of `get_grad`'s return value
            should equal (num_params,N,N), where the return value's
            i-th element is the matrix derivative of the unitary
            with respect to the i-th parameter.

            The default implementation raises `NotImplementedError`; gates
            that support differentiation should override this method. Use
            :func:`is_differentiable` to check whether a gate has done so.
        """
        raise NotImplementedError(
            f'{self.name} does not have a gradient definition.',
        )

    def get_unitary_and_grad(
        self,
        params: RealVector = [],
    ) -> tuple[UnitaryMatrix, npt.NDArray[np.complex128]]:
        """
        Return a tuple combining the outputs of `get_unitary` and `get_grad`.

        Args:
            params (RealVector): The unitary parameters, see
                :func:`Unitary.get_unitary` for more info.

        Returns:
            tuple: tuple containing:
                UnitaryMatrix: The unitary matrix, see
                :func:`Unitary.get_unitary` for more info.

                np.ndarray: The unitary's gradient, see :func:`get_grad`.

        Notes:
            Can be overridden to potentially speed up optimization by
            calculating both at the same time.
        """
        return (self.get_unitary(params), self.get_grad(params))

    def is_differentiable(self) -> bool:
        """Return true if this gate has a gradient definition."""
        return type(self).get_grad is not Gate.get_grad

    @property
    def qasm_name(self) -> str:
        """The qasm identifier for this gate."""
        if not self.is_qubit_only():
            raise AttributeError('QASM only supports qubit gates.')

        return getattr(self, '_qasm_name')

    def get_qasm_gate_def(self) -> str:
        """Returns a qasm gate definition block for this gate."""
        if not self.is_qubit_only():
            raise AttributeError('QASM only supports qubit gates.')

        return ''

    def get_qasm(self, params: RealVector, location: CircuitLocation) -> str:
        """Returns the qasm string for this gate."""
        return '{}({}) q[{}];\n'.format(
            self.qasm_name,
            ', '.join([str(p) for p in params]),
            '], q['.join([str(q) for q in location]),
        ).replace('()', '')

    def get_inverse_params(self, params: RealVector = []) -> RealVector:
        """
        Return the parameters that invert the gate.

        Args:
            params (RealVector): The parameters of the gate to invert.

        Note:
            - The default implementation returns the same paramters because
              the default implementation of `Gate.get_inverse` returns a
              :class:`DaggerGate` wrapper of the gate. The wrapper will
              correctly handle the inversion. When overriding `get_inverse`,
              on a parameterized gate this method should be overridden
              as well.
        """
        return params

    def get_inverse(self) -> Gate:
        """Return the gate's inverse as a gate."""
        if self.is_constant() and self.is_self_inverse():
            return self

        from bqskit.ir.gates.composed import DaggerGate
        return getattr(self, '_inverse', DaggerGate(self))
        # TODO: Fill out inverse definitions throughout the gate library

    with_frozen_params: ClassVar[
        Callable[[Gate, dict[int, float]], FrozenParameterGate]
    ]
    with_all_frozen_params: ClassVar[
        Callable[[Gate, list[float]], FrozenParameterGate]
    ]

    def __repr__(self) -> str:
        return self.name
