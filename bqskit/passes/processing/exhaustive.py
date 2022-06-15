"""This module implements the ExhaustiveGateRemovalPass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.structure import CircuitStructure
from bqskit.utils.typing import is_real_number
_logger = logging.getLogger(__name__)


class ExhaustiveGateRemovalPass(BasePass):
    """
    The ExhaustiveGateRemovalPass class.

    Use instantiation to remove the most possible gates from the circuit.
    """

    def __init__(
        self,
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
        collection_filter: Callable[[Operation], bool] | None = None,
        scoring_fn: Callable[[Circuit], float] | None = None,
    ) -> None:
        """
        Construct a ExhaustiveGateRemovalPass.

        Args:

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-10)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

            collection_filter (Callable[[Operation], bool] | None):
                A predicate that determines which operations should be
                attempted to be removed. Called with each operation
                in the circuit. If this returns true, this pass will
                attempt to remove that operation. Defaults to all
                operations.

            scoring_fn (Callable[[Circuit], float]):
                A scoring function for the circuits to determine which one
                to select. Defaults to gate counts weighted by their size.
        """

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.collection_filter = collection_filter or default_collection_filter

        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                ' collection_filter, got %s.' % type(self.collection_filter),
            )

        self.scoring_fn = scoring_fn or default_scoring_fn

        if not callable(self.scoring_fn):
            raise TypeError(
                'Expected callable method that maps Circuits to floats for'
                ' scoring_fn, got %s.' % type(self.scoring_fn),
            )

        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
        }
        self.instantiate_options.update(instantiate_options)

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Starting exhaustive gate removal.')

        target = self.get_target(circuit, data)

        # Frontier tracks circuits successfully instantiated to target
        frontier = [circuit.copy()]

        # Track best cicuit seen so far
        best_circuit = None
        best_score = -np.inf

        # Keep removing until no more successful circuits
        while len(frontier) > 0:

            # Expand each element of frontier by removing gates
            expanded_circuits = []

            # Don't repeat circuit structures
            circuits_seen = set()

            for c in frontier:
                for cycle, op in c.operations_with_cycles():
                    point = (cycle, op.location[0])
                    copy = c.copy()
                    copy.pop(point)
                    structure = CircuitStructure(copy)
                    if structure not in circuits_seen:
                        expanded_circuits.append(copy)
                        circuits_seen.add(structure)

            # Instantiate them all
            instantiated_circuits = self.execute(
                data,
                Circuit.instantiate,
                expanded_circuits,
                target=target,
                **self.instantiate_options,
            )

            # Process them
            next_frontier = []
            for c in instantiated_circuits:
                if self.cost(c, target) < self.success_threshold:
                    next_frontier.append(c)

                    score = self.scoring_fn(c)
                    if score > best_score:
                        best_circuit = c
                        best_score = score

            frontier = next_frontier

        # Keep best circuit if one found
        if best_circuit is not None:
            circuit.become(best_circuit)


def default_collection_filter(op: Operation) -> bool:
    return True


def default_scoring_fn(circuit: Circuit) -> float:
    """Default scoring function."""
    score = 0.0
    for op in circuit:
        score -= (op.num_qudits - 1) * 100 + 1
    return score
