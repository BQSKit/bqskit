"""This module implements the QFASTDecompositionPass class."""
from __future__ import annotations

import copy
import logging
from typing import Any
from typing import Sequence

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import PauliGate
from bqskit.ir.gates.composed.vlg import VariableLocationGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost import CostFunctionGenerator
from bqskit.ir.opt.cost import HilbertSchmidtResidualsGenerator
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

_logger = logging.getLogger(__name__)


class QFASTDecompositionPass(SynthesisPass):
    """
    A pass performing one round of decomposition from the QFAST algorithm.

    References:
        E. Younis, K. Sen, K. Yelick and C. Iancu, "QFAST: Conflating Search
        and Numerical Optimization for Scalable Quantum Circuit Synthesis,"
        2021 IEEE International Conference on Quantum Computing and
        Engineering (QCE), 2021, pp. 232-243, doi: 10.1109/QCE52317.2021.00041.
    """

    def __init__(
        self,
        gate: Gate = PauliGate(2),
        success_threshold: float = 1e-10,
        progress_threshold: float = 5e-3,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_depth: int | None = None,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        QFASTDecompositionPass Constructor.

        Args:
            gate (Gate): The gate to decompose unitaries into. Ensure
                that the gate specified is expressive over the unitaries
                being synthesized. (Default: PauliGate(2))

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
            ValueError: If max_depth is nonpositive.
        """

        if not isinstance(gate, Gate):
            raise TypeError('Expected gate to be a Gate, got %s' % type(gate))

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not is_real_number(progress_threshold):
            raise TypeError(
                'Expected real number for progress_threshold'
                ', got %s' % type(progress_threshold),
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

        self.gate = gate
        self.success_threshold = success_threshold
        self.progress_threshold = progress_threshold
        self.cost = cost
        self.max_depth = max_depth
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': self.cost,
            'method': 'minimization',
        }
        self.instantiate_options.update(instantiate_options)

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""

        # Skip any unitaries too small for the configured gate.
        if self.gate.num_qudits > utry.num_qudits:
            _logger.warning('Skipping unitary synthesis since gate is larger.')
            return Circuit.from_unitary(utry)

        # Create empty circuit with same size and radixes as `utry`.
        circuit = Circuit(utry.num_qudits, utry.radixes)

        # Calculate relevant coupling_graph and create the VLG head.
        model = self.get_model(utry, data)
        locations = model.get_locations(self.gate.num_qudits)
        vlg_head = VariableLocationGate(self.gate, locations, circuit.radixes)
        circuit.append_gate(vlg_head, list(range(utry.num_qudits)))

        # Track depth and distances
        depth = 1
        last_dist = 1.0
        failed_locs: list[tuple[CircuitLocation, float]] = []

        # Main loop
        while True:

            # Instantiate circuit
            circuit = self.execute(
                data,
                Circuit.instantiate,
                [circuit],
                target=utry,
                **self.instantiate_options,
            )[0]

            dist = self.cost.calc_cost(circuit, utry)
            _logger.info(f'Instantiated depth {depth} at {dist} cost.')

            if dist < self.success_threshold:
                self.finalize(circuit, utry, data)
                _logger.info('Successful synthesis.')
                return circuit

            # Expand or restrict head
            location = self.get_location_of_head(circuit)

            if last_dist - dist >= self.progress_threshold:
                _logger.info('Progress has been made, depth increasing.')
                last_dist = dist
                self.expand(circuit, location, locations)
                depth += 1
                failed_locs = []

            elif not self.can_restrict(circuit[-1, 0]):
                _logger.info('Progress has not been made.')
                _logger.info('Cannot restrict further, depth increasing.')
                failed_locs.append((location, dist))
                failed_locs.sort(key=lambda x: x[1])
                location, last_dist = failed_locs[0]
                self.expand(circuit, location, locations)
                depth += 1
                failed_locs = []

            else:
                _logger.info('Progress has not been made, restricting model.')
                failed_locs.append((location, dist))
                self.restrict_head(circuit, location)

    def get_location_of_head(self, circuit: Circuit) -> CircuitLocation:
        """Return the current location of the `circuit`'s VLG head."""
        head_gate: VariableLocationGate = circuit[-1, 0].gate  # type: ignore
        return CircuitLocation(head_gate.get_location(circuit[-1, 0].params))

    def finalize(
        self,
        circuit: Circuit,
        utry: UnitaryMatrix,
        data: dict[str, Any],
    ) -> None:
        """Finalize the circuit by replacing the head with self.gate."""
        # Replace Head with self.gate
        location = self.get_location_of_head(circuit)
        _logger.info(f'Final gate added at location {location}.')
        circuit.pop()
        circuit.append(Operation(self.gate, location))

        # Reinstantiate
        dist = self.cost.calc_cost(circuit, utry)
        while dist > self.success_threshold:
            circuit.become(
                self.execute(
                    data,
                    Circuit.instantiate,
                    [circuit],
                    target=utry,
                    **(self.instantiate_options),
                )[0],
            )
            dist = self.cost.calc_cost(circuit, utry)

        _logger.info(f'Final circuit found with cost: {dist}.')

    def expand(
        self,
        circuit: Circuit,
        location: CircuitLocation,
        locations: Sequence[CircuitLocation],
    ) -> None:
        """Expand the circuit after a successful layer."""
        _logger.info(f'Expanding by adding a gate on qubits {location}.')
        circuit.insert(-1, Operation(self.gate, location))
        self.lift_head_restrictions(circuit, locations)
        self.restrict_head(circuit, location)

    def can_restrict(self, head: Operation) -> bool:
        """Return true if the VLG head can be restricted further."""
        head_gate: VariableLocationGate = head.gate  # type: ignore
        return len(head_gate.locations) > 1

    def restrict_head(
        self,
        circuit: Circuit,
        location: CircuitLocation,
    ) -> None:
        """
        Remove `location` from the VLG Head in `circuit`.

        Args:
            circuit (Circuit): The circuit to restrict its VLG head.

            location (CircuitLocation): The location to remove from the
                VLG head.
        """
        _logger.debug(f'Removing location {location} from the VLG head.')
        head_gate: VariableLocationGate = circuit[-1, 0].gate  # type: ignore
        locations = copy.deepcopy(head_gate.locations)
        locations.remove(location)
        new_vlg = VariableLocationGate(self.gate, locations, circuit.radixes)
        new_head = Operation(new_vlg, list(range(circuit.num_qudits)))
        circuit.pop()
        circuit.append(new_head)

    def lift_head_restrictions(
        self,
        circuit: Circuit,
        locations: Sequence[CircuitLocation],
    ) -> None:
        """Set the `circuit`'s VLG head's valid locations to `locations`."""
        _logger.debug('Lifting restrictions.')
        _logger.debug(f'Reset VLG head location pool to {locations}.')
        vlg_gate = VariableLocationGate(self.gate, locations, circuit.radixes)
        new_head = Operation(vlg_gate, list(range(circuit.num_qudits)))
        circuit.pop()
        circuit.append(new_head)
