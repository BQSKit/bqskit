"""This module implements the QFactor class."""
from __future__ import annotations

import logging
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from bqskit.ir.opt.instantiater import Instantiater
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class QFactor(Instantiater):
    """The QFactor circuit instantiater."""

    def __init__(
        self,
        diff_tol_a: float = 1e-12,
        diff_tol_r: float = 1e-4,
        dist_tol: float = 1e-10,
        max_iters: int = 1000,
        min_iters: int = 0,
        slowdown_factor: float = 0.0,
        reinit_delay: int = 40,
        log_delay: int = 100,
        **kwargs: Any,
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

            reinit_delay (int): The number of iterations in between
                circuit tensor reinitializations. The circuit tensor is
                reinitialized every so-often to avoid numerical drift.
                Smaller values increase runtime, larger values led to a
                greater potential for numerical inaccuracy. (Default: 40)

            log_delay (int): The number of iterations in between log
                messages. (Default: 100)
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
                'Expected nonnegative integer for min_iters'
                ', got %d.' % min_iters,
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

        if not is_integer(reinit_delay):
            raise TypeError(
                'Expected int for reinit_delay, got %s.' % type(reinit_delay),
            )

        if reinit_delay <= 0:
            raise ValueError(
                'Expected nonnegative integer for reinit_delay'
                ', got %d.' % reinit_delay,
            )

        if not is_integer(log_delay):
            raise TypeError(
                'Expected int for log_delay, got %s.' % type(log_delay),
            )

        if log_delay <= 0:
            raise ValueError(
                'Expected nonnegative integer for log_delay'
                ', got %d.' % log_delay,
            )

        self.diff_tol_a = diff_tol_a
        self.diff_tol_r = diff_tol_r
        self.dist_tol = dist_tol
        self.max_iters = max_iters
        self.min_iters = min_iters
        self.slowdown_factor = slowdown_factor
        self.reinit_delay = reinit_delay
        self.log_delay = log_delay

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

        typed_target: UnitaryMatrix

        # Make a copy to preserve original
        circuit = circuit.copy()

        circuit.set_params(x0)
        ct = self.initialize_circuit_tensor(circuit, typed_target)

        c1 = 0
        c2 = 1
        it = 0

        while True:

            # Termination conditions
            if it > self.min_iters:

                diff_tol = self.diff_tol_a + self.diff_tol_r * np.abs(c1)
                if np.abs(c1 - c2) <= diff_tol:
                    diff = np.abs(c1 - c2)
                    _logger.info(
                        f'Terminated: |c1 - c2| = {diff}'
                        ' <= diff_tol_a + diff_tol_r * |c1|.',
                    )
                    break

                if it > self.max_iters:
                    _logger.info('Terminated: iteration limit reached.')
                    break

            it += 1

            self.sweep_circuit(ct, circuit)

            c2 = c1
            c1 = np.abs(np.trace(ct.get_unitary().get_numpy()))
            c1 = 1 - (c1 / (2 ** ct.get_size()))

            if c1 <= self.dist_tol:
                _logger.info(f'Terminated: c1 = {c1} <= dist_tol.')
                return circuit.get_params()

            if it % self.log_delay == 0:
                _logger.debug(f'iteration: {it}, cost: {c1}')

            if it % self.reinit_delay == 0:
                ct = self.initialize_circuit_tensor(circuit, typed_target)

        return circuit.get_params()

    def initialize_circuit_tensor(
        self,
        circuit: Circuit,
        target: UnitaryMatrix,
    ) -> UnitaryBuilder:
        ct = UnitaryBuilder(circuit.get_size(), circuit.get_radixes())
        ct.apply_right(target.get_dagger(), list(range(circuit.get_size())))
        ct.apply_right(circuit.get_unitary(), list(range(circuit.get_size())))
        return ct

    def sweep_circuit(
        self,
        ct: UnitaryBuilder,
        circuit: Circuit,
    ) -> None:
        """Perform a QFactor optimization sweep from right to left and back."""
        # from right to left
        for op in reversed(circuit):

            # Remove current gate from right of circuit tensor
            ct.apply_right(op.get_unitary(), op.location, inverse=True)

            # Update current gate
            if op.gate.is_parameterized():
                env = ct.calc_env_matrix(op.location)
                op.params = op.gate.optimize(env)  # type: ignore
                # TODO: slowdown_factor

            # Add updated gate to left of circuit tensor
            ct.apply_left(op.get_unitary(), op.location)

        # from left to right
        for op in circuit:

            # Remove current gate from left of circuit tensor
            ct.apply_left(op.get_unitary(), op.location, inverse=True)

            # Update current gate
            if op.gate.is_parameterized():
                env = ct.calc_env_matrix(op.location)
                op.params = op.gate.optimize(env)  # type: ignore
                # TODO: slowdown_factor

            # Add updated gate to right of circuit tensor
            ct.apply_right(op.get_unitary(), op.location)

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
