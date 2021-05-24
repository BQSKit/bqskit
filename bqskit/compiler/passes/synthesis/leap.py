"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import linregress

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
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        layer_generator: LayerGenerator = SimpleLayerGenerator(),
        success_threshold: float = 1e-6,
        cost: CostFunctionGenerator = HilbertSchmidtGenerator(),
        max_layer: int | None = None,
        min_prefix_size: int = 3,
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
                described by the cost function. (Default: 1e-6)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                 to unlimited. (Default: None)

            min_prefix_size (int): The minimum number of layers needed
                to prefix the circuit.

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

            kwargs (dict[str, Any]): Keyword arguments that are passed
                directly to SynthesisPass's constructor. See SynthesisPass
                for more info.

        Raises:
            ValueError: If `max_depth` or `min_prefix_size` is nonpositive.
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

        if min_prefix_size is not None and not is_integer(min_prefix_size):
            raise TypeError(
                'Expected min_prefix_size to be an integer, got %s'
                % type(min_prefix_size),
            )

        if min_prefix_size is not None and min_prefix_size <= 0:
            raise ValueError(
                'Expected min_prefix_size to be positive, got %d.'
                % int(min_prefix_size),
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
        self.min_prefix_size = min_prefix_size
        self.instantiate_options: dict[str, Any] = {'cost_fn_gen': self.cost}
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
        best_dists = [best_dist]
        best_layers = [0]
        last_prefix_layer = 0
        _logger.info('Search started, initial layer has cost: %e.' % best_dist)

        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            # Generate successors and evaluate each
            for circuit in self.layer_gen.gen_successors(top_circuit, data):

                circuit.instantiate(utry, **self.instantiate_options)

                dist = self.cost.calc_cost(circuit, utry)

                if dist < best_dist:
                    _logger.info(
                        'New best circuit found with %d layer%s and cost: %e.'
                        % (layer + 1, '' if layer == 0 else 's', dist),
                    )
                    best_dist = dist
                    best_circ = circuit
                    best_layer = layer

                    if self.check_leap_condition(
                        layer + 1,
                        best_dist,
                        best_layers,
                        best_dists,
                        last_prefix_layer,
                    ):
                        _logger.info(
                            'Prefix formed at %d layers.' % (layer + 1),
                        )
                        last_prefix_layer = layer + 1
                        frontier.clear()
                        if self.max_layer is None or layer + 1 < self.max_layer:
                            frontier.add(circuit, layer + 1)
                        break

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

    def check_leap_condition(
        self,
        new_layer: int,
        best_dist: float,
        best_layers: list[int],
        best_dists: list[float],
        last_prefix_layer: int,
    ) -> bool:
        """
        Return true if the leap condition is satisfied.

        Args:
            new_layer (int): The current layer in search.

            best_dist (float): The current best distance in search.

            best_layers (list[int]): The list of layers associated
                with recorded best distances.

            best_dists (list[float]): The list of recorded best
                distances.

            last_prefix_layer (int): The last layer a prefix was formed.
        """

        with np.errstate(invalid='ignore', divide='ignore'):
            # Calculate predicted best value
            m, y_int, _, _, _ = linregress(best_layers, best_dists)
        predicted_best = m * (new_layer) + y_int

        # Track new values
        best_layers.append(new_layer)
        best_dists.append(best_dist)

        if np.isnan(predicted_best):
            return False

        # Compute difference between actual value
        delta = predicted_best - best_dist

        _logger.debug(
            'Predicted best value %f for new best best with delta %f.'
            % (predicted_best, delta),
        )

        layers_added = new_layer - last_prefix_layer
        return delta < 0 and layers_added >= self.min_prefix_size
