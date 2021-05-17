"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.passes.synthesispass import SynthesisPass
from bqskit.compiler.search.frontier import Frontier
from bqskit.compiler.search.generator import LayerGenerator
from bqskit.compiler.search.heuristic import HeuristicFunction
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions.hilbertschmidt import HilbertSchmidtGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number


_logger = logging.getLogger(__name__)


class LEAPSynthesisPass(SynthesisPass):
    """
    The LEAPSynthesisPass class.

    References:
        Smith, Ethan, et al. “LEAP: Scaling Numerical Optimization Based
        Synthesis Using an Incremental Approach.” International Workshop
        of Quantum Computing Software at Supercomputing (2020).
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction,
        layer_generator: LayerGenerator,
        success_threshold: float = 1e-6,
        cost: CostFunctionGenerator = HilbertSchmidtGenerator(),
        max_depth: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            layer_generator (LayerGenerator): The successor function
                to guide node expansion.

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-6)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_depth (int): The maximum number of gates to append without
                success before termination. If left as None it will default
                 to unlimited. (Default: None)

            kwargs (dict[str, Any]): Keyword arguments that are passed
                directly to SynthesisPass's constructor. See SynthesisPass
                for more info.

        Raises:
            ValueError: If max_depth is nonpositive.
        """
        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        if not isinstance(layer_generator, LayerGenerator):
            raise TypeError(
                'Expected LayerGenerator, got %s.'
                % type(layer_generator),
            )

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

        if max_depth is not None and not is_integer(max_depth):
            raise TypeError(
                'Expected max_depth to be an integer, got %s' % type(
                    max_depth,
                ),
            )

        if max_depth is not None and max_depth <= 0:
            raise ValueError(
                'Expected max_depth to be positive, got %d.' % int(max_depth),
            )

        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_depth = max_depth
        super().__init__(**kwargs)

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry` into a circuit, see SynthesisPass for more info."""
        frontier = Frontier(utry, self.heuristic_function)

        best_dist = 1.0
        best_circ = None

        self.layer_gen.gen_initial_layer(utry, data)

        while not frontier.empty():
            child_circuits = self.layer_gen.gen_successors(frontier.pop(), data)
            for circuit in child_circuits:
                if self.max_depth and circuit.get_num_cycles() < self.max_depth:
                    continue

                circuit.instantiate(utry, cost_fn_gen=self.cost)

                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold:
                    _logger.info('Circuit found with cost: %e.' % dist)
                    _logger.info('Successful synthesis.')
                    return circuit

                if dist < best_dist:
                    _logger.info('Circuit found with cost: %e.' % dist)
                    best_dist = dist
                    best_circ = circuit

                # TODO: Add LEAP ALGORITHM

                frontier.add(circuit)

        _logger.info('Frontier emptied.')
        _logger.info('Returning best known circuit with dist: %e.' % best_dist)

        if best_circ is None:
            _logger.warning('No circuit found during search.')
            best_circ = Circuit.from_unitary(utry)

        return best_circ
