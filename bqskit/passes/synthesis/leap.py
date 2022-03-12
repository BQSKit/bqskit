"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from dask.distributed import get_client
from scipy.stats import linregress

from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators import SimpleLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import AStarHeuristic
from bqskit.passes.synthesis.synthesis import SynthesisPass
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
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        min_prefix_size: int = 3,
        instantiate_options: dict[str, Any] = {},
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

            store_partial_solutions (bool): Whether to store partial solutions
                at different depths inside of the data dict. (Default: False)

            partials_per_depth (int): The maximum number of partials
                to store per search depth. No effect if
                `store_partial_solutions` is False. (Default: 25)

            min_prefix_size (int): The minimum number of layers needed
                to prefix the circuit.

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

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
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        frontier = Frontier(utry, self.heuristic_function)
        data['window_markers'] = []

        # Seed the search with an initial layer
        initial_layer = self.layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(utry, **self.instantiate_options)
        frontier.add(initial_layer, 0)

        # Track best circuit, initially the initial layer
        leap_data: dict[str, Any] = {}
        leap_data['best_dist'] = self.cost.calc_cost(initial_layer, utry)
        leap_data['best_circ'] = initial_layer
        leap_data['best_layer'] = 0
        leap_data['best_dists'] = [leap_data['best_dist']]
        leap_data['best_layers'] = [0]
        leap_data['last_prefix_layer'] = 0
        if self.store_partial_solutions:
            leap_data['psols'] = {}
        _logger.info(
            'Search started, initial layer has cost: %e.' %
            leap_data['best_dist'],
        )

        if leap_data['best_dist'] < self.success_threshold:
            _logger.info('Successful synthesis.')
            return initial_layer

        if 'executor' in data:  # In Parallel
            client = get_client()
            while not frontier.empty():
                top_circuit, layer = frontier.pop()

                # Generate successors and evaluate each
                successors = self.layer_gen.gen_successors(top_circuit, data)

                # Submit instantiate jobs
                futures = self.batched_instantiate(
                    successors,
                    utry,
                    client,
                    **self.instantiate_options,
                )

                # Wait for and gather results
                circuits = self.gather_best_results(
                    futures,
                    client,
                    self.cost.calc_cost,
                    utry,
                )

                for circuit in circuits:
                    if self.evaluate_node(
                        circuit,
                        utry,
                        data,
                        frontier,
                        layer,
                        leap_data,
                    ):
                        if self.store_partial_solutions:
                            data['psols'] = leap_data['psols']
                        return circuit

        else:  # Sequentially
            while not frontier.empty():
                top_circuit, layer = frontier.pop()
                # Generate successors and evaluate each
                successors = self.layer_gen.gen_successors(top_circuit, data)

                for circuit in successors:
                    circuit.instantiate(utry, **self.instantiate_options)
                    if self.evaluate_node(
                        circuit,
                        utry,
                        data,
                        frontier,
                        layer,
                        leap_data,
                    ):
                        if self.store_partial_solutions:
                            data['psols'] = leap_data['psols']
                        return circuit

        _logger.info('Frontier emptied.')
        _logger.info(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (
                leap_data['best_layer'], '' if leap_data['best_layer'] == 1
                else 's', leap_data['best_dist'],
            ),
        )
        if self.store_partial_solutions:
            data['psols'] = leap_data['psols']

        return leap_data['best_circ']

    def evaluate_node(
        self,
        circuit: Circuit,
        utry: UnitaryMatrix,
        data: dict[str, Any],
        frontier: Frontier,
        layer: int,
        leap_data: dict[str, Any],
    ) -> bool:
        dist = self.cost.calc_cost(circuit, utry)

        if dist < self.success_threshold:
            _logger.info('Successful synthesis.')
            return True

        if self.check_new_best(
            layer + 1,
            dist,
            leap_data['best_layer'],
            leap_data['best_dist'],
        ):
            _logger.info(
                'New best circuit found with %d layer%s and cost: %e.'
                % (layer + 1, '' if layer == 0 else 's', dist),
            )
            leap_data['best_dist'] = dist
            leap_data['best_circ'] = circuit
            leap_data['best_layer'] = layer + 1

            if self.check_leap_condition(layer + 1, leap_data):
                _logger.info('Prefix formed at %d layers.' % (layer + 1))
                leap_data['last_prefix_layer'] = layer + 1
                frontier.clear()
                data['window_markers'].append(circuit.num_cycles)
                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(circuit, layer + 1)

        if self.store_partial_solutions:
            if layer not in leap_data['psols']:
                leap_data['psols'][layer] = []

            leap_data['psols'][layer].append((circuit.copy(), dist))

            if len(leap_data['psols'][layer]) > self.partials_per_depth:
                leap_data['psols'][layer].sort(key=lambda x: x[1])
                del leap_data['psols'][layer][-1]

        if self.max_layer is None or layer + 1 < self.max_layer:
            frontier.add(circuit, layer + 1)
        return False

    def check_new_best(
        self,
        layer: int,
        dist: float,
        best_layer: int,
        best_dist: float,
    ) -> bool:
        """
        Check if the new layer depth and dist are a new best node.

        Args:
            layer (int): The current layer in search.

            dist (float): The current distance in search.

            best_layer (int): The current best layer in the search tree.

            best_dist (float): The current best distance in search.
        """
        better_layer = (
            dist < best_dist
            and (
                best_dist >= self.success_threshold
                or layer <= best_layer
            )
        )
        better_dist_and_layer = (
            dist < self.success_threshold and layer < best_layer
        )
        return better_layer or better_dist_and_layer

    def check_leap_condition(
        self,
        new_layer: int,
        leap_data: dict[str, Any],
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
            m, y_int, _, _, _ = linregress(
                leap_data['best_layers'], leap_data['best_dists'],
            )
        predicted_best = m * (new_layer) + y_int

        # Track new values
        leap_data['best_layers'].append(new_layer)
        leap_data['best_dists'].append(leap_data['best_dist'])

        if np.isnan(predicted_best):
            return False

        # Compute difference between actual value
        delta = predicted_best - leap_data['best_dist']

        _logger.debug(
            'Predicted best value %f for new best best with delta %f.'
            % (predicted_best, delta),
        )

        layers_added = new_layer - leap_data['last_prefix_layer']
        return delta < 0 and layers_added >= self.min_prefix_size
