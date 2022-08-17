"""This module implements the QPredictDecompositionPass class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.opt.cost import CostFunctionGenerator
from bqskit.ir.opt.cost import HilbertSchmidtResidualsGenerator
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.math import pauli_expansion
from bqskit.utils.math import unitary_log_no_i
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

_logger = logging.getLogger(__name__)


class QPredictDecompositionPass(SynthesisPass):
    """
    The QPredictDecompositionPass class.

    Breaks down the input unitary into blocks.
    """

    def __init__(
        self,
        block_size_start: int = 2,
        block_size_limit: int | None = None,
        fail_limit: int = 3,
        success_threshold: float = 1e-8,
        progress_threshold_r: float = 5e-2,
        progress_threshold_a: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_depth: int | None = None,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        QPredictDecompositionPass Constructor.

        Args:
            block_size_start (int): The smallest block size to append
                each step. (Default: 2)

            block_size_limit (int | None): The largest block size to append
                each step. If left as None, the unitary being synthesized
                provides the limit. (Default: None)

            fail_limit (int): The amount of tries to make progress before
                increasing block size. (Default: 2)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-6)

            progress_threshold (float): The distance necessary to improve
                for the synthesis algorithm to complete a layer and move
                on. Lowering this will led to synthesis going deeper quicker,
                and raising it will force to algorithm to spend more time
                on each layer. Caution, changing this too much might break
                the synthesis algorithm. (Default: 5e-3)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_depth (int): The maximum number of gates to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: If `block_size_start` is nonpositive.

            ValueError: If `block_size_limit` is less than `block_size_start`.

            ValueError: If `fail_limit` is nonpositive.

            ValueError: If `max_depth` is nonpositive.
        """

        if not is_integer(block_size_start):
            raise TypeError(
                'Expected block_size_start to be an integer, got %s'
                % type(block_size_start),
            )

        if block_size_start <= 0:
            raise ValueError(
                'Expected block_size_start to be positive, got %d.'
                % block_size_start,
            )

        if block_size_limit is not None and not is_integer(block_size_limit):
            raise TypeError(
                'Expected block_size_limit to be an integer, got %s'
                % type(block_size_limit),
            )

        if block_size_limit is not None and block_size_limit < block_size_start:
            raise ValueError(
                'Expected block_size_limit to be larger than block_size_start,'
                'got %d.' % block_size_limit,
            )

        if not is_integer(fail_limit):
            raise TypeError(
                'Expected fail_limit to be an integer, got %s'
                % type(fail_limit),
            )

        if fail_limit <= 0:
            raise ValueError(
                'Expected fail_limit to be positive, got %d.' % fail_limit,
            )

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not is_real_number(progress_threshold_r):
            raise TypeError(
                'Expected real number for progress_threshold_r'
                ', got %s' % type(progress_threshold_r),
            )

        if not is_real_number(progress_threshold_a):
            raise TypeError(
                'Expected real number for progress_threshold_a'
                ', got %s' % type(progress_threshold_a),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if max_depth is not None and not is_integer(max_depth):
            raise TypeError(
                'Expected max_depth to be an integer, got %s'
                % type(max_depth),
            )

        if max_depth is not None and max_depth <= 0:
            raise ValueError(
                'Expected max_depth to be positive, got %d.' % int(max_depth),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.block_size_start = block_size_start
        self.block_size_limit = block_size_limit
        self.fail_limit = fail_limit
        self.success_threshold = success_threshold
        self.progress_threshold_r = progress_threshold_r
        self.progress_threshold_a = progress_threshold_a
        self.cost = cost
        self.max_depth = max_depth
        self.instantiate_options = {
            'min_iters': 25,
            'diff_tol_r': 1e-4,
        }
        self.instantiate_options.update(instantiate_options)

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""

        # 0. Skip any unitaries too small for the configured block.
        if self.block_size_start >= utry.num_qudits:
            _logger.warning(
                'Skipping synthesis: block size is too large.',
            )
            return Circuit.from_unitary(utry)

        # 1. Create empty circuit with same size and radixes as `utry`.
        circuit = Circuit(utry.num_qudits, utry.radixes)

        # 2. Calculate block sizes
        block_size_end = utry.num_qudits - 1
        if self.block_size_limit is not None:
            block_size_end = self.block_size_limit

        # 3. Calculate relevant coupling_graphs
        model = self.get_model(utry, data)
        locations = [
            model.get_locations(i)
            for i in range(self.block_size_start, block_size_end + 1)
        ]

        # 3. Bottom-up synthesis: build circuit up one gate at a time
        layer = 1
        last_cost = 1.0
        last_loc = None

        while True:
            remainder = utry.dagger @ circuit.get_unitary()
            sorted_locations = self.analyze_remainder(remainder, locations)

            for loc in sorted_locations:

                # Never predict the previous location
                if loc == last_loc:
                    continue

                _logger.info(f'Trying next predicted location {loc}.')
                circuit.append_gate(VariableUnitaryGate(len(loc)), loc)
                circuit.instantiate(
                    utry,
                    **self.instantiate_options,  # type: ignore
                )
                cost = self.cost.calc_cost(circuit, utry)
                _logger.info(f'Instantiated; layers: {layer}, cost: {cost:e}.')

                if cost < self.success_threshold:
                    _logger.info(f'Circuit found with cost: {cost:e}.')
                    _logger.info('Successful synthesis.')
                    return circuit

                progress_threshold = self.progress_threshold_a
                progress_threshold += self.progress_threshold_r * np.abs(cost)
                if last_cost - cost >= progress_threshold:
                    _logger.info('Progress has been made, depth increasing.')
                    last_loc = loc
                    last_cost = cost
                    layer += 1
                    break

                _logger.info('Progress has not been made.')
                circuit.pop((-1, loc[0]))

    def analyze_remainder(
        self,
        R: UnitaryMatrix,
        locations: Sequence[Sequence[CircuitLocation]],
    ) -> list[CircuitLocation]:
        """
        Perform remainder analysis on `R` to sort `locations`.

        Args:
            R (UnitaryMatrix): The remainder to analyze.

            locations (Sequence[Sequence[CircuitLocation]]): List of locations
                grouped by block size.

        Returns:
            (list[CircuitLocation]): Sorted list of locations for next block
                based on remainder analysis.
        """
        _logger.info('Performing remainder analysis.')
        pauli_coefs = pauli_expansion(unitary_log_no_i(R))  # type: ignore

        locations_by_index: dict[int, set[CircuitLocation]] = {}
        for location_group in locations:
            for loc in location_group:
                for qudit_index in loc:
                    if qudit_index not in locations_by_index:
                        locations_by_index[qudit_index] = set()

                    locations_by_index[qudit_index].add(loc)

        location_scores_by_size: dict[int, dict[CircuitLocation, float]] = {}
        for location_group in locations:
            for loc in location_group:
                if len(loc) not in location_scores_by_size:
                    location_scores_by_size[len(loc)] = {}
                location_scores_by_size[len(loc)][loc] = 0.0

        for coef_idx, coef in enumerate(pauli_coefs):
            for qudit_index in self.decode_qubits(coef_idx):
                for loc in locations_by_index[qudit_index]:
                    location_scores_by_size[len(loc)][loc] += np.abs(coef)

        sorted_locations_by_size: dict[int, list[CircuitLocation]] = {
            size:
            sorted(
                list(location_scores.keys()),
                key=lambda x: location_scores[x],
                reverse=True,
            )
            for size, location_scores in location_scores_by_size.items()
        }

        sorted_locations_length: dict[int, int] = {
            size: len(sorted_locations)
            for size, sorted_locations in sorted_locations_by_size.items()
        }

        sorted_locations_by_sorted_size = sorted(
            sorted_locations_by_size.items(),
            key=lambda x: x[0],
        )

        sorted_locations: list[CircuitLocation] = []
        for size, locations in sorted_locations_by_sorted_size:
            for i in range(self.fail_limit):
                if i < sorted_locations_length[size]:
                    sorted_locations.append(locations[i])

        sorted_locations.extend(sorted_locations_by_sorted_size[-1][1])

        _logger.debug('Found order of locations:')
        _logger.debug(sorted_locations)
        return sorted_locations

    def decode_qubits(self, pauli_coef_index: int) -> list[int]:
        """Decode `pauli_coef_index` into qubit indices."""
        qubit_indices = []
        qudit_index = 0
        while pauli_coef_index > 0:
            if ((pauli_coef_index & 3) >> 1) | (pauli_coef_index & 1):
                qubit_indices.append(qudit_index)
            pauli_coef_index >>= 2
            qudit_index += 1
        return qubit_indices
