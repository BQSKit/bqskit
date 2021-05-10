"""This module implements the QFactor class."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bqskit.ir.opt.instantiater import Instantiater
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


class QFactor(Instantiater):
    """The QFactor circuit instantiater."""

    def __init__(
        self,
        diff_tol_a: float = 1e-12,
        diff_tol_r: float = 1e-6,
        dist_tol: float = 1e-10,
        max_iters: int = 100000,
        min_iters: int = 1000,
        slowdown_factor: float = 0.0,
    ) -> None:
        """
        Construct and configure a QFactor Instantiater.

        Args:
            diff_tol_a (float): Terminate when the difference in cost
                between iterations is less than this threshold.
                (Default: 1e-12)

            diff_tol_r (float): Terminate when the relative difference in
                cost between iterations is less than this threshold:
                    |c1 - c2| <= diff_tol_a + diff_tol_r * abs( c1 )
                (Default: 1e-6)

            dist_tol (float): Terminate successfully when the cost is
                less than this threshold. (Default: 1e-10)

            max_iters (int): Maximum number of iterations.
                (Default: 100000)

            min_iters (int): Minimum number of iterations.
                (Default: 1000)

            slowdown_factor (float): A positive number less than 1.
                The larger this factor, the slower the optimization.
                Increasing this may increase runtime and reduce chance
                of getting stuck in local minima.
                (Default: 0.0)
        """

        if not is_real_number(diff_tol_a):
            raise TypeError(
                'Expected float for diff_tol_a, got %s.' % type(diff_tol_a),
            )

        if diff_tol_a <= 0 or diff_tol_a >= 0.5:
            raise ValueError(
                'Expected 0 < diff_tol_a < 0.5, got %d.' % diff_tol_a,
            )

        if not is_real_number(diff_tol_r):
            raise TypeError(
                'Expected float for diff_tol_r, got %s.' % type(diff_tol_r),
            )

        if diff_tol_r <= 0 or diff_tol_r >= 0.5:
            raise ValueError(
                'Expected 0 < diff_tol_r < 0.5, got %d.' % diff_tol_r,
            )

        if not is_real_number(dist_tol):
            raise TypeError(
                'Expected float for dist_tol, got %s.' % type(dist_tol),
            )

        if dist_tol <= 0 or dist_tol >= 1:
            raise ValueError(
                'Expected 0 < dist_tol < 1, got %d.' % dist_tol,
            )

        if not is_integer(max_iters):
            raise TypeError(
                'Expected int for max_iters, got %s.' % type(max_iters),
            )

        if max_iters < 0:
            raise ValueError(
                'Expected positive integer for max_iters, got %d.' % max_iters,
            )

        if not is_integer(min_iters):
            raise TypeError(
                'Expected int for min_iters, got %s.' % type(min_iters),
            )

        if min_iters < 0:
            raise ValueError(
                'Expected positive integer for min_iters, got %d.' % min_iters,
            )

        if not is_real_number(slowdown_factor):
            raise TypeError(
                'Expected float for slowdown_factor, got %s.'
                % type(slowdown_factor),
            )

        if slowdown_factor < 0 or slowdown_factor >= 1:
            raise ValueError(
                'Expected 0 <= slowdown_factor < 1, got %d.' % slowdown_factor,
            )

        self.diff_tol_a = diff_tol_a
        self.diff_tol_r = diff_tol_r
        self.dist_tol = dist_tol
        self.max_iters = max_iters
        self.min_iters = min_iters
        self.slowdown_factor = slowdown_factor

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
        x0: np.ndarray,
    ) -> np.ndarray:
        """Instantiate `circuit`, see Instantiater for more info."""
        typed_target = self.check_target(target)

        if isinstance(typed_target, StateVector):
            raise NotImplementedError(
                'QFactor is not currently implemented for StateVector targets.',
            )

        return x0  # TODO

    @staticmethod
    def is_capable(circuit: Circuit) -> bool:
        """Return true if the circuit can be instantiated."""
        return all(
            isinstance(gate, LocallyOptimizableUnitary)
            for gate in circuit.get_gate_set()
        )

    @staticmethod
    def get_violation_report(circuit: Circuit) -> str:
        """
        Return a message explaining why `circuit` cannot be instantiated.

        Args:
            circuit (Circuit): Generate a report for this circuit.

        Raises:
            ValueError: If `circuit` can be instantiated with this
                instantiater.
        """

        invalid_gates = {
            gate
            for gate in circuit.get_gate_set()
            if not isinstance(gate, LocallyOptimizableUnitary)
        }

        if len(invalid_gates) == 0:
            raise ValueError('Circuit can be instantiated.')

        return (
            'Cannot instantiate circuit with qfactor'
            ' because the following gates are not locally optimizable: %s.'
            % ', '.join(str(g) for g in invalid_gates)
        )

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'qfactor'
