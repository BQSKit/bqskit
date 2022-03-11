"""This module implements the QFASTDecompositionPass class."""
from __future__ import annotations

import copy
import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.machine import MachineModel
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
    The QFASTDecompositionPass class.

    Performs one QFAST decomposition step breaking down a unitary into a
    sequence of smaller operations.
    """

    def __init__(
        self,
        gate: Gate = PauliGate(2),
        success_threshold: float = 1e-6,
        progress_threshold: float = 5e-3,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_depth: int | None = None,
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

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""

        # 0. Skip any unitaries too small for the configured gate.
        if self.gate.num_qudits > utry.num_qudits:
            _logger.warning('Skipping unitary synthesis since gate is larger.')
            return Circuit.from_unitary(utry)

        # 1. Create empty circuit with same size and radixes as `utry`.
        circuit = Circuit(utry.num_qudits, utry.radixes)

        # 2. Calculate relevant coupling_graph and create the VLG head.
        # If a MachineModel is provided in the data dict, it will be used.
        # Otherwise all-to-all connectivity is assumed.
        model = None

        if 'machine_model' in data:
            model = data['machine_model']

        if (
            not isinstance(model, MachineModel)
            or model.num_qudits < circuit.num_qudits
        ):
            _logger.warning(
                'MachineModel not specified or invalid;'
                ' defaulting to all-to-all.',
            )
            model = MachineModel(circuit.num_qudits)
        locations = model.get_locations(self.gate.num_qudits)
        circuit.append_gate(
            VariableLocationGate(self.gate, locations, circuit.radixes),
            list(range(utry.num_qudits)),
        )

        # 3. Bottom-up synthesis: build circuit up one gate at a time
        depth = 1
        last_dist = 1.0
        failed_locs: list[tuple[CircuitLocation, float]] = []

        while True:
            circuit.instantiate(utry, cost_fn_gen=self.cost)

            dist = self.cost.calc_cost(circuit, utry)

            _logger.info(
                'Finished optimizing depth %d at %e cost.' % (depth, dist),
            )

            if dist < self.success_threshold:
                _logger.info('Circuit found with cost: %e.' % dist)
                _logger.info('Successful synthesis.')
                self.finalize(utry, circuit)
                return circuit

            location = self.get_location_of_head(circuit)

            if last_dist - dist >= self.progress_threshold:
                _logger.info('Progress has been made, depth increasing.')
                last_dist = dist
                self.expand(circuit, location, locations)
                depth += 1

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

    def expand(
        self,
        circuit: Circuit,
        location: CircuitLocation,
        locations: Sequence[CircuitLocation],
    ) -> None:
        """Expand the circuit after a successful layer."""

        _logger.info(
            'Expanding circuit by adding a gate at location %s.' % str(
                location,
            ),
        )

        circuit.insert(-1, Operation(self.gate, location))
        self.lift_head_restrictions(circuit, locations)
        self.restrict_head(circuit, location)

    def finalize(self, utry: UnitaryMatrix, circuit: Circuit) -> None:
        """Finalize the circuit."""
        location = self.get_location_of_head(circuit)
        _logger.info('Final gate added at location %s.' % str(location))

        circuit.insert(-1, Operation(self.gate, location))
        circuit.pop()

        # TODO: reconsider method arg to instantiate call
        circuit.instantiate(utry, cost_fn_gen=self.cost, method='minimization')

        dist = self.cost.gen_cost(circuit, utry).get_cost(circuit.params)
        _logger.info('Final circuit distance: %e.' % dist)

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

        _logger.debug(
            'Removing location %s from the VLG head.' % str(location),
        )

        head_gate: VariableLocationGate = circuit[-1, 0].gate  # type: ignore
        locations = copy.deepcopy(head_gate.locations)
        locations.remove(location)
        new_head = Operation(
            VariableLocationGate(self.gate, locations, circuit.radixes),
            list(range(circuit.num_qudits)),
        )
        circuit.pop()
        circuit.append(new_head)

    def lift_head_restrictions(
        self,
        circuit: Circuit,
        locations: Sequence[CircuitLocation],
    ) -> None:
        """Set the `circuit`'s VLG head's valid locations to `locations`."""

        _logger.debug(
            'Lifting restrictions, setting VLG head location pool to %s.'
            % str(locations),
        )

        new_head = Operation(
            VariableLocationGate(self.gate, locations, circuit.radixes),
            list(range(circuit.num_qudits)),
        )

        circuit.pop()
        circuit.append(new_head)
