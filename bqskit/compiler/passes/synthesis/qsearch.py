"""This module implements the QSearchSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.passes.synthesispass import SynthesisPass
from bqskit.compiler.search.frontier import Frontier
from bqskit.compiler.search.generator import LayerGenerator
from bqskit.compiler.search.generators import SimpleLayerGenerator
from bqskit.compiler.search.heuristic import HeuristicFunction
from bqskit.compiler.search.heuristics import AStarHeuristic
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions.hilbertschmidt import HilbertSchmidtGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number


_logger = logging.getLogger(__name__)


class QSearchSynthesisPass(SynthesisPass):
    """
    The QSearchSynthesisPass class.

    References:
        Davis, Marc G., et al. “Towards Optimal Topology Aware Quantum
        Circuit Synthesis.” 2020 IEEE International Conference on Quantum
        Computing and Engineering (QCE). IEEE, 2020.
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        layer_generator: LayerGenerator = SimpleLayerGenerator(),
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtGenerator(),
        max_layer: int | None = None,
        instantiate_options: dict[str, Any] = {},
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
                described by the cost function. (Default: 1e-10)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                 to unlimited. (Default: None)

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

            kwargs (dict[str, Any]): Keyword arguments that are passed
                directly to SynthesisPass's constructor. See SynthesisPass
                for more info.

        Raises:
            ValueError: If `max_depth` is nonpositive.
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

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                'Expected max_layer to be an integer, got %s' % type(max_layer),
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                'Expected max_layer to be positive, got %d.' % int(max_layer),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.instantiate_options = {'cost_fn_gen': self.cost}
        self.instantiate_options.update(instantiate_options)
        super().__init__(**kwargs)

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry` into a circuit, see SynthesisPass for more info."""
        frontier = Frontier(utry, self.heuristic_function)

        # Seed the search with an initial layer
        initial_layer = self.layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(
            utry,
            **self.instantiate_options,  # type: ignore
        )
        frontier.add(initial_layer, 0)

        # Track best circuit, initially the initial layer
        best_dist = self.cost.calc_cost(initial_layer, utry)
        best_circ = initial_layer
        best_layer = 0
        _logger.info('Search started, initial layer has cost: %e.' % best_dist)

        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            # Generate successors and evaluate each
            for circuit in self.layer_gen.gen_successors(top_circuit, data):

                circuit.instantiate(
                    utry,
                    **self.instantiate_options,  # type: ignore
                )

                dist = self.cost.calc_cost(circuit, utry)

                if dist < best_dist:
                    _logger.info(
                        'New best circuit found with %d layer%s and cost: %e.'
                        % (layer + 1, '' if layer == 0 else 's', dist),
                    )
                    best_dist = dist
                    best_circ = circuit
                    best_layer = layer

                if dist < self.success_threshold:
                    _logger.info('Successful synthesis.')
                    return circuit

                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(circuit, layer + 1)

        _logger.info('Frontier emptied.')
        _logger.info(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )

        return best_circ
