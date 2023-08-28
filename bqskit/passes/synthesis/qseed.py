"""This module implements the QSeedSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any, Sequence

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators import SeedLayerGenerator
from bqskit.passes.search.generators import SimpleLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import AStarHeuristic
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number


_logger = logging.getLogger(__name__)


class QSeedSynthesisPass(SynthesisPass):
    """
    A pass implementing the seeded QSearch synthesis algorithm.

    References:
        Davis, Marc G., et al. “Towards Optimal Topology Aware Quantum
        Circuit Synthesis.” 2020 IEEE International Conference on Quantum
        Computing and Engineering (QCE). IEEE, 2020.
    """

    def __init__(
        self,
        seeds: Circuit | Sequence[Circuit],
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        forward_generator: LayerGenerator = SimpleLayerGenerator(),
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            seeds (Circuit | Sequence[Circuit]): Circuit or Circuits from which 
                to start looking for solutions.

            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            forward_generator (LayerGenerator): The successor function
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

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: If `max_depth` is nonpositive.
        """
        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                f'Expected HeursiticFunction, got {type(heuristic_function)}'
            )

        if not isinstance(forward_generator, LayerGenerator):
            raise TypeError(
                f'Expected LayerGenerator, got {type(forward_generator)}'
            )

        if not is_real_number(success_threshold):
            raise TypeError(
                f'Expected real number for success_threshold'
                f', got {type(success_threshold)}'
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                f'Expected cost to be a CostFunctionGenerator, got {type(cost)}'
            )

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                f'Expected max_layer to be an integer, got {type(max_layer)}'
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                f'Expected max_layer to be positive, got {int(max_layer)}'
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got '
                f'{type(instantiate_options)}'
            )

        self.seeds = [seeds] if not isinstance(seeds, Sequence) else list(seeds)
        
        if not all([isinstance(c, Circuit) for c in self.seeds]):
            raise TypeError(
                'Expected Circuit or Sequence of Circuits for `seed`.'
            )

        self.seed_dim = self.seeds[0].dim

        if not all([s.dim == self.seed_dim for s in self.seeds]):
            raise TypeError('Expected seeds to be the same dimension')

        self.heuristic_function = heuristic_function
        self.forward_generator = forward_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': self.cost,
            'method': 'minimization',
        }
        self.instantiate_options.update(instantiate_options)
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth

        # For scenarios where blocks with width different from seeds
        self.mismatch_layer_generator = SimpleLayerGenerator()

    async def synthesize(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""

        frontier = Frontier(utry, self.heuristic_function)
        # PRNG seed
        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        # Seed the search with an initial layer
        # Check to see if a new seed has been provided
        seeds: list[Circuit] = self.seeds
        if 'seed_circuits' in data:
            seeds = data['seed_circuits']
            if not all([isinstance(s, Circuit) for s in seeds]):
                raise TypeError(
                    'Seeds passed through the `data` dict must be '
                    'provided as a list of Circuits.'
                )

        if not self._check_same_dims(utry.dim, seeds):
            _logger.debug(
                'Seed and unitary dimensions do not match '
                f'({self.seed_dim} != {utry.dim}). '
                'Using mis-match layer generator.'
            )
            layer_gen = self.mismatch_layer_generator
        else:
            layer_gen = SeedLayerGenerator(seeds, self.forward_generator)

        initial_layers: list[Circuit]|Circuit = layer_gen.gen_initial_layer(
            utry, data
        )
        if not isinstance(initial_layers, list):
            initial_layers = [initial_layers]

        # Evaluate seeds
        initial_layers = await get_runtime().map(
            Circuit.instantiate,
            initial_layers,
            target=utry,
            **instantiate_options,
        )
        for initial_layer in initial_layers:
            frontier.add(initial_layer, 0)

        # Track best circuit
        best_dist: float = 1.1 # Higher than highest possible distance
        best_circ: Circuit|None = None
        for initial_layer in initial_layers:
            dist = self.cost.calc_cost(initial_layer, utry)
            if dist < best_dist:
                best_dist = dist
                best_circ = initial_layer
        best_layer = 0

        # Track partial solutions
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layers
        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis.')
            return best_circ 

        # Main loop
        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            # Generate successors
            successors = layer_gen.gen_successors(top_circuit, data)

            # Instantiate successors
            successor_circuits = await get_runtime().map(
                Circuit.instantiate,
                successors,
                target=utry,
                **instantiate_options,
            )

            # Evaluate successors
            for circuit in successor_circuits:
                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold:
                    _logger.debug('Successful synthesis.')
                    if self.store_partial_solutions:
                        data['psols'] = psols
                    return circuit

                if dist < best_dist:
                    _logger.debug(
                        'New best circuit found with %d layer%s and cost: %e.'
                        % (layer + 1, '' if layer == 0 else 's', dist),
                    )
                    best_dist = dist
                    best_circ = circuit
                    best_layer = layer

                if self.store_partial_solutions:
                    if layer not in psols:
                        psols[layer] = []

                    psols[layer].append((circuit.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(circuit, layer + 1)

        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )
        if self.store_partial_solutions:
            data['psols'] = psols

        return best_circ
    
    def _check_same_dims(
            self, 
            target_dim: int, 
            circuit_list: Sequence[Circuit],
        ) -> bool:
        """
        Whether or not all circuits in the list have the same `dim` as
        `target_dim`.

        Args:
            target_dim (int): Dimension size to check for.

            circuit_list (Sequence[Circuit]): List of circuits to check.

        Returns:
            (bool): If all circuits in `circuit_list` have the same dim
                as `target_dim` returns True. Otherwise returns False.
        """
        return all([c.dim == target_dim for c in circuit_list])
