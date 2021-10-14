"""This module implements the QSearchSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

from dask.distributed import as_completed
from dask.distributed import worker_client

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


class QSearchSynthesisPass(SynthesisPass):
    """
    A pass implementing the QSearch A* synthesis algorithm.

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
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
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

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

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
        self.instantiate_options: dict[str, Any] = {'cost_fn_gen': self.cost}
        self.instantiate_options.update(instantiate_options)

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        frontier = Frontier(utry, self.heuristic_function)

        instantiate_options = self.instantiate_options
        if 'executor' in data:
            instantiate_options['parallel'] = True

        # Seed the search with an initial layer
        initial_layer = self.layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(utry, **instantiate_options)
        frontier.add(initial_layer, 0)

        # Track best circuit, initially the initial layer
        search_data: dict[str, Any] = {}
        search_data['best_dist'] = self.cost.calc_cost(initial_layer, utry)
        search_data['best_circ'] = initial_layer
        search_data['best_layer'] = 0
        _logger.info(
            'Search started, initial layer has cost: %e.' %
            search_data['best_dist'],
        )

        if search_data['best_dist'] < self.success_threshold:
            _logger.info('Successful synthesis.')
            return initial_layer

        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            # Generate successors and evaluate each
            successors = self.layer_gen.gen_successors(top_circuit, data)

            if 'executor' in data:  # In Parallel
                with worker_client() as client:
                    futures = client.map(
                        Circuit.instantiate,
                        successors,
                        pure=False,
                        target=utry,
                        **instantiate_options,
                    )
                    for _, circuit in as_completed(futures, with_results=True):
                        if self.evaluate_node(
                            circuit,
                            utry,
                            data,
                            frontier,
                            layer,
                            search_data,
                        ):
                            for future in futures:
                                if not future.done:
                                    client.cancel(future)
                            return circuit

            else:  # Sequentially
                for circuit in successors:
                    circuit.instantiate(utry, **instantiate_options)
                    if self.evaluate_node(
                        circuit,
                        utry,
                        data,
                        frontier,
                        layer,
                        search_data,
                    ):
                        return circuit

        _logger.info('Frontier emptied.')
        _logger.info(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (
                search_data['best_layer'], '' if search_data['best_layer'] == 1
                else 's', search_data['best_dist'],
            ),
        )

        return search_data['best_circ']

    def evaluate_node(
        self,
        circuit: Circuit,
        utry: UnitaryMatrix,
        data: dict[str, Any],
        frontier: Frontier,
        layer: int,
        search_data: dict[str, Any],
    ) -> bool:
        dist = self.cost.calc_cost(circuit, utry)

        if dist < self.success_threshold:
            _logger.info('Successful synthesis.')
            return True

        if dist < search_data['best_dist']:
            _logger.info(
                'New best circuit found with %d layer%s and cost: %e.'
                % (layer + 1, '' if layer == 0 else 's', dist),
            )
            search_data['best_dist'] = dist
            search_data['best_circ'] = circuit
            search_data['best_layer'] = layer

        if self.max_layer is None or layer + 1 < self.max_layer:
            frontier.add(circuit, layer + 1)

        return False
