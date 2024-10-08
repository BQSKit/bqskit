from __future__ import annotations

import logging
from enum import Enum
from typing import Any
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.gateset import GateSetLike
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.operation import Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes import SynthesisPass
from bqskit.passes.search.generators import WideLayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
# from bqskit.passes import QSearchSynthesisPass

_logger = logging.getLogger(__name__)


class DecompositionOrder(Enum):
    LeftToRight = 1
    RightToLeft = 2
    TowardsMiddle = 3


class CircuitWideAwareLayerGenerator(WideLayerGenerator):

    def __init__(
            self,
            full_circuit: Circuit,
            cycle_to_grow: int,
            location_to_grow: CircuitLocation,
            multi_qudit_gates: GateSetLike = VariableUnitaryGate(3),
            single_qudit_gate: Gate | None = VariableUnitaryGate(1),
    ):

        super().__init__(multi_qudit_gates, single_qudit_gate)

        if not isinstance(full_circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(full_circuit))

        if not is_integer(cycle_to_grow):
            raise TypeError(
                'Expected integer for cycle_to_grow, got %s.'
                % type(cycle_to_grow),
            )
        # TODO: add type check for the location_to_grow variable

        self.full_circuit = full_circuit
        self.cycle_to_grow = cycle_to_grow
        self.location_to_grow = location_to_grow

    def gen_initial_layer(
            self,
            target: UnitaryMatrix | StateVector | StateSystem,
            data: PassData,
    ) -> Circuit:
        initial_circuit_to_grow = super().gen_initial_layer(target, data)

        if self.single_qudit_gate is not None:
            self.full_circuit.insert_circuit(
                self.cycle_to_grow,
                initial_circuit_to_grow,
                self.location_to_grow,
            )

        return self.full_circuit

    def gen_successors(self, circuit: Circuit, data: PassData) -> list[Circuit]:

        if not isinstance(circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(circuit))

        if circuit.num_qudits < 2:
            raise ValueError('Cannot expand a single-qudit circuit.')

        # Get the coupling graph of the location to grow
        cg = data.model.coupling_graph.get_subgraph(self.location_to_grow)

        # Generate successors
        successors = []
        for mg in self.multi_qudit_gates:
            if mg.num_qudits > len(self.location_to_grow):
                continue
            for loc in cg.get_subgraphs_of_size(mg.num_qudits):
                translated_loc = [self.location_to_grow[i] for i in loc]

                successor = circuit.copy()
                successor.insert_gate(self.cycle_to_grow, mg, translated_loc)
                if self.single_qudit_gate is not None:
                    for q in translated_loc:
                        successor.insert_gate(
                            self.cycle_to_grow, self.single_qudit_gate, q,
                        )
                successors.append(successor)

        return successors


class FullCircuitAwareIndividualDecomposition(BasePass):

    def __init__(
        self,
        synthesis_pass_class: type[SynthesisPass],
        # synthesis_pass_class: type[SynthesisPass] = QSearchSynthesisPass,
        decompositionOrder: DecompositionOrder = DecompositionOrder.LeftToRight,
        multi_qudit_gates: GateSetLike = VariableUnitaryGate(3),
        single_qudit_gate: Gate | None = VariableUnitaryGate(1),
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
        collection_filter: Callable[[Operation], bool] | None = None,
        synthesis_options: dict[str, Any] = {},

    ):

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

        if not isinstance(synthesis_options, dict):
            raise TypeError(
                'Expected dictionary for synthesis_options, got %s.'
                % type(synthesis_options),
            )

        if not issubclass(synthesis_pass_class, SynthesisPass):
            raise TypeError(
                'Expected type subclass of SynthesisPass for '
                'synthesis_pass_class, got %s.' % type(synthesis_pass_class),
            )

        self.collection_filter = collection_filter or default_collection_filter

        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                ' collection_filter, got %s.' % type(self.collection_filter),
            )

        self.decompositionOrder = decompositionOrder
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)

        self.synthesis_options = synthesis_options
        self.synthesis_pass_class = synthesis_pass_class

        if isinstance(multi_qudit_gates, Gate):
            multi_qudit_gates = [multi_qudit_gates]

        if not all(isinstance(g, Gate) for g in multi_qudit_gates):
            raise TypeError(
                'Expected gate for multi_qudit_gates, got %s.'
                % [type(g) for g in multi_qudit_gates],
            )

        if single_qudit_gate is not None:
            if not isinstance(single_qudit_gate, Gate):
                raise TypeError(
                    'Expected gate for single_qudit_gate, got %s.'
                    % type(single_qudit_gate),
                )

            if single_qudit_gate.num_qudits != 1:
                raise ValueError(
                    'Expected single-qudit gate'
                    ', got a gate that acts on %d qudits.'
                    % single_qudit_gate.num_qudits,
                )

            sr = single_qudit_gate.radixes[0]
            for mg in multi_qudit_gates:
                if any(r != sr for r in mg.radixes):
                    raise ValueError(
                        'Radix mismatch between gates'
                        f': {mg.radixes} !~ {single_qudit_gate.radixes}.',
                    )

        self.multi_qudit_gates: list[Gate] = list(multi_qudit_gates)
        self.single_qudit_gate = single_qudit_gate

    async def run(self, circuit: Circuit, data: PassData) -> None:

        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        grow_from_left = self.decompositionOrder in [
            DecompositionOrder.LeftToRight, DecompositionOrder.TowardsMiddle,
        ]
        should_alternate = (
            self.decompositionOrder == DecompositionOrder.TowardsMiddle
        )

        _logger.debug(
            f'Starting FullCircuitAwareIndividualDecomposition with '
            f'{self.decompositionOrder}, {should_alternate = } '
            f'{grow_from_left = }',
        )

        for cycle, op in circuit.operations_with_cycles():
            if self.collection_filter(op):
                first_cycle_to_decompose_on_the_left = cycle
                break

        for cycle, op in circuit.operations_with_cycles(reverse=True):
            if self.collection_filter(op):
                first_cycle_to_decompose_on_the_right = cycle
                break

        cycle_in_circuit = [
            first_cycle_to_decompose_on_the_right,
            first_cycle_to_decompose_on_the_left,
        ]

        # Iterate over the operations to decompose
        more_cycles_to_grow = True
        iteration = 0
        while (more_cycles_to_grow):

            iteration += 1
            current_cycle_index = cycle_in_circuit[grow_from_left]
            current_cycle = circuit[current_cycle_index]
            assert len(current_cycle) == 1
            current_location = current_cycle[0].location

            _logger.debug(
                f'Will remove operation {current_cycle[0]} in cycle '
                f'{current_cycle_index} at  location {current_location}, and '
                f' decompose it to {self.multi_qudit_gates}',
            )

            circuit.pop_cycle(current_cycle_index)

            orig_number_of_cycles = len(circuit)

            layer_gen = CircuitWideAwareLayerGenerator(
                circuit,
                current_cycle_index,
                current_location,
                self.multi_qudit_gates,
                self.single_qudit_gate,
            )

            self.synthesis_options['layer_generator'] = layer_gen

            self.synthesis_options['instantiate_options'] = \
                self.instantiate_options

            synthesis_pass = self.synthesis_pass_class(
                **self.synthesis_options,
            )

            await synthesis_pass.run(circuit, data)

            new_number_of_cycles = len(circuit)
            cycles_add = new_number_of_cycles - orig_number_of_cycles

            dist = self.cost(circuit, data.target)

            _logger.debug(
                f'The decomposing in iteration {iteration} added {cycles_add}'
                f' gates and the current dist is {dist:.6e}.',
            )

            step = 1 if grow_from_left else -1
            next_cycle_to_grow = current_cycle_index + \
                (cycles_add if grow_from_left else -1)

            _logger.debug(
                f'{new_number_of_cycles = } {next_cycle_to_grow = } {step = }',
            )

            # Find the next gate to decompose
            while (
                next_cycle_to_grow < new_number_of_cycles
                or next_cycle_to_grow >= 0
            ):

                if self.collection_filter(circuit[next_cycle_to_grow][0]):
                    break  # We found the gate to decompose
                next_cycle_to_grow += step
            else:
                _logger.debug('Will stop as reached end.')
                more_cycles_to_grow = False

            cycle_in_circuit[grow_from_left] = next_cycle_to_grow

            # Need to push the right pointer with the number of cycles added.
            # TODO: consider using pointers to actual gates and not cycle
            # numbers to avoid this.
            if grow_from_left:
                cycle_in_circuit[False] += cycles_add

            # Making sure that when the direction is towards the middle,
            # that the two pointers don't pass each other
            if cycle_in_circuit[True] > cycle_in_circuit[False]:
                more_cycles_to_grow = False

            if should_alternate:
                grow_from_left = not grow_from_left

        _logger.debug(
            f'Decomposition finished after {iteration} iterations'
            f' with distance {dist:.6e}.',
        )


def default_collection_filter(op: Operation) -> bool:
    return True
