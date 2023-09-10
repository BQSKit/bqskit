"""This module implements the Rebase2QuditGatePass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.point import CircuitPoint as Point
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from bqskit.utils.typing import is_sequence
_logger = logging.getLogger(__name__)


class Rebase2QuditGatePass(BasePass):
    """
    The Rebase2QuditGatePass class.

    Will use instantiation to change the a 2-qudit gate to a different one.
    """

    def __init__(
        self,
        gate_in_circuit: Gate | Sequence[Gate],
        new_gate: Gate | Sequence[Gate],
        max_depth: int = 3,
        max_retries: int = -1,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
        single_qudit_gate: Gate = U3Gate(),
    ) -> None:
        """
        Construct a Rebase2QuditGatePass.

        Args:
            gate_in_circuit (Gate | Sequence[Gate]): The two-qudit gate
                or gates in the circuit that you want to replace.

            new_gate (Gate | Sequence[Gate]): The two-qudit new gate or
                gates you want to put in the circuit.

            max_depth (int): The maximum number of new gates to replace
                an old gate with. (Default: 3)

            max_retries (int): The number of retries for the same gate
                before we increase the maximum depth. If left as -1,
                then never increase max depth. (Default: -1)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

            single_qudit_gate (Gate): A single-qudit gate to fill
                in between two-qudit gates.

        Raises:
            ValueError: If `gate_in_circuit` or `new_gate` is not a 2-qudit
                gate.

            ValueError: if `max_depth` is nonnegative.
        """

        if is_sequence(gate_in_circuit):
            if any(not isinstance(g, Gate) for g in gate_in_circuit):
                raise TypeError('Expected Gate or Gate list.')

        elif not isinstance(gate_in_circuit, Gate):
            raise TypeError(f'Expected Gate, got {type(gate_in_circuit)}.')

        else:
            gate_in_circuit = [gate_in_circuit]

        if any(g.num_qudits != 2 for g in gate_in_circuit):
            raise ValueError('Expected 2-qudit gate.')

        if is_sequence(new_gate):
            if any(not isinstance(g, Gate) for g in new_gate):
                raise TypeError('Expected Gate or Gate list.')

        elif not isinstance(new_gate, Gate):
            raise TypeError(f'Expected Gate, got {type(new_gate)}.')

        else:
            new_gate = [new_gate]

        if any(g.num_qudits != 2 for g in new_gate):
            raise ValueError('Expected 2-qudit gate.')

        if not is_integer(max_depth):
            raise TypeError(f'Expected integer, got {max_depth}.')

        if max_depth < 0:
            raise ValueError(f'Expected nonnegative depth, got: {max_depth}.')

        if not is_integer(max_retries):
            raise TypeError(f'Expected integer, got {max_retries}.')

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

        if not isinstance(single_qudit_gate, Gate):
            raise TypeError(f'Expected Gate, got {type(single_qudit_gate)}.')

        if single_qudit_gate.num_qudits != 1:
            raise ValueError(
                f'Expected single-qudit gate, got {single_qudit_gate}.',
            )

        self.gates: list[Gate] = list(gate_in_circuit)
        self.ngates = new_gate
        self.max_depth = max_depth  # TODO: Compute from weyl chamber coords
        self.max_retries = max_retries
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
        }
        self.instantiate_options.update(instantiate_options)
        self.sq = single_qudit_gate
        temps = self.generate_new_gate_templates(self.ngates, self.sq)
        self.circs, self.counts, self.overdrive, msgs = temps
        self.replaced_log_messages = msgs

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        instantiate_options = self.instantiate_options.copy()
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed
        _logger.debug(f'Rebasing gates from {self.gates} to {self.ngates}.')

        target = self.get_target(circuit, data)

        for g in self.gates:
            # Track retries to check for no progress
            num_retries = 0
            prev_count = circuit.count(g)

            while g in circuit.gate_set:
                # Change the seed every iteration to prevent stalls
                if instantiate_options['seed'] is not None:
                    instantiate_options['seed'] += 1

                # Check if we made progress from last loop
                gates_left = circuit.count(g)
                if prev_count == gates_left:
                    num_retries += 1
                else:
                    prev_count = gates_left
                    num_retries = 0

                # Group together a 2-qubit block composed of gates from old set
                point = self.group_near_gates(
                    circuit,
                    circuit.point(g),
                    self.gates,
                )
                circuits_with_new_gate = []
                for circ in self.circs:
                    circuit_copy = circuit.copy()
                    circuit_copy.replace_with_circuit(point, circ)
                    circuits_with_new_gate.append(circuit_copy)

                # If we have exceeded the number of retries, up the max depth
                if self.max_retries >= 0 and num_retries > self.max_retries:
                    _logger.debug('Exceeded max retries, increasing depth.')
                    circuit_copy = circuit.copy()
                    circuit_copy.replace_with_circuit(point, self.overdrive)
                    circuits_with_new_gate.append(circuit_copy)

                instantiated_circuits = await get_runtime().map(
                    Circuit.instantiate,
                    circuits_with_new_gate,
                    target=target,
                    **instantiate_options,
                )

                dists = [self.cost(c, target) for c in instantiated_circuits]

                # Find the successful circuit with the least gates
                best_index = None
                best_count = self.max_depth + 2
                for i, dist in enumerate(dists):
                    if dist < self.success_threshold:
                        if self.counts[i] < best_count:
                            best_index = i
                            best_count = self.counts[i]

                if best_index is None:
                    circuit.unfold(point)
                    continue

                _logger.debug(self.replaced_log_messages[best_index])
                circuit.become(instantiated_circuits[best_index])

    def group_near_gates(
        self,
        circuit: Circuit,
        center: Point,
        gates: list[Gate],
    ) -> Point:
        """Group gates similar to the gate at center on the same qubits."""
        op = circuit[center]
        qubits = op.location
        counts = {g: 0.0 for g in gates}
        counts[op.gate] += 1.0

        # Go to the left until cant
        i = 0
        moving_left = True
        while moving_left:
            i += 1
            if center.cycle - i < 0:
                i = center.cycle
                break
            for q in qubits:
                point = (center.cycle - i, q)
                if not circuit.is_point_idle(point):
                    lop = circuit[point]
                    if any(p not in qubits for p in lop.location):
                        i -= 1
                        moving_left = False
                        break
                    if lop.num_qudits != 1 and lop.gate not in gates:
                        i -= 1
                        moving_left = False
                        break
                    if lop.num_qudits == 2:
                        counts[lop.gate] += 0.5

        j = 0
        moving_right = True
        while moving_right:
            j += 1
            if center.cycle + j >= circuit.num_cycles:
                j = circuit.num_cycles - center.cycle - 1
                break
            for q in qubits:
                point = (center.cycle + j, q)
                if not circuit.is_point_idle(point):
                    rop = circuit[point]
                    if any(p not in qubits for p in rop.location):
                        j -= 1
                        moving_right = False
                        break
                    if rop.num_qudits != 1 and rop.gate not in gates:
                        j -= 1
                        moving_right = False
                        break
                    if rop.num_qudits == 2:
                        counts[rop.gate] += 0.5

        region = {q: (center.cycle - i, center.cycle + j) for q in qubits}
        grouped_gate_str = ', '.join([
            f'{int(c)} {g}' + ('s' if c > 1 else '')
            for g, c in counts.items()
        ])
        _logger.debug(f'Grouped together {grouped_gate_str}.')
        return circuit.fold(region)

    def generate_new_gate_templates(
        self,
        new_gates: Sequence[Gate],
        single_qudit_gate: Gate,
    ) -> tuple[list[Circuit], list[int], Circuit, list[str]]:
        """Generate the templates to be instantiated over old circuits."""
        circs: list[Circuit] = []
        counts: list[int] = []

        circ = Circuit(2, new_gates[0].radixes)
        circ.append_gate(single_qudit_gate, 0)
        circ.append_gate(single_qudit_gate, 1)
        circs.append(circ)
        counts.append(0)

        for g in new_gates:
            for i in range(1, self.max_depth + 1):
                circ = Circuit(2, new_gates[0].radixes)
                circ.append_gate(single_qudit_gate, 0)
                circ.append_gate(single_qudit_gate, 1)

                for _ in range(i):
                    circ.append_gate(g, (0, 1))
                    circ.append_gate(single_qudit_gate, 0)
                    circ.append_gate(single_qudit_gate, 1)

                counts.append(i)
                circs.append(circ)

        # Add overdrive circuit, incase we exceed retry limit
        overdrive = circs[-1].copy()
        overdrive.append_gate(new_gates[-1], (0, 1))
        overdrive.append_gate(single_qudit_gate, 0)
        overdrive.append_gate(single_qudit_gate, 1)
        counts.append(counts[-1] + 1)

        # Preprocess log messages
        replaced_log_messages = []
        for circ in circs + [overdrive]:
            g_counts = [circ.count(g) for g in circ.gate_set]
            gate_count_str = ', '.join([
                f'{c} {g}' + ('s' if c > 1 else '')
                for c, g in zip(g_counts, circ.gate_set)
            ])
            msg = f'Replaced gate with {gate_count_str}.'
            replaced_log_messages.append(msg)

        return circs, counts, overdrive, replaced_log_messages
