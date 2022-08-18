"""This module implements the GeneralizedSabreAlgorithm class."""
from __future__ import annotations

import copy
import logging

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import SwapGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.qis.graph import CouplingGraph


_logger = logging.getLogger(__name__)


class GeneralizedSabreAlgorithm():
    """
    Implements methods for Sabre-based layout and routing algorithms using a
    modified heuristic to accommodate larger than 2-qudit gates.

    References:
        Gushu Li, Yufei Ding, and Yuan Xie. 2019. Tackling the Qubit
        Mapping Problem for NISQ-Era Quantum Devices. In Proceedings of
        the 24th ACM International Conference on Architectural
        Support for Programming Languages and Operating Systems
        (ASPLOS 2019). Association for Computing Machinery, New York, NY,
        USA, 1001-1014. https://doi.org/10.1145/3297858.3304023

        Casey Duckering, Jonathan M. Baker, Andrew Litteken, and Frederic
        T. Chong. 2021. Orchestrated trios: compiling for efficient
        communication in Quantum programs with 3-Qubit gates. In Proceedings
        of the 26th ACM International Conference on Architectural Support
        for Programming Languages and Operating Systems (ASPLOS 2021).
        Association for Computing Machinery, New York, NY, USA, 375-385.
        https://doi.org/10.1145/3445814.3446718
    """

    def __init__(
        self,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
    ) -> None:
        """
        Construct a GeneralizedSabreAlgorithm.

        Args:
            decay_delta (float): The amount to adjust the decay factor by
                each time a swap is applied. Set to zero to disable decay.
                (Default: 0.001)

            decay_reset_interval (int): The amount of swaps to apply before
                reseting the decay factors. (Default: 5)

            decay_reset_on_gate (bool): If true, reset decay factors when
                a logical gate is applied. (Default: True)

            extended_set_size (int): The size of the look-ahead or extended
                set. Set to zero to disable look ahead. (Default: 20)

            extended_set_weight (float): The weight on the extended set
                term when scoring potential swaps. (Default: 0.5)
        """
        if not isinstance(decay_delta, float):
            raise TypeError(
                'Expected float for decay_delta'
                f', got {type(decay_delta)}',
            )

        if not isinstance(decay_reset_interval, int):
            raise TypeError(
                'Expected int for decay_reset_interval'
                f', got {type(decay_reset_interval)}',
            )

        if not isinstance(decay_reset_on_gate, bool):
            raise TypeError(
                'Expected bool for decay_reset_on_gate'
                f', got {type(decay_reset_on_gate)}',
            )

        if not isinstance(extended_set_size, int):
            raise TypeError(
                'Expected int for extended_set_size'
                f', got {type(extended_set_size)}',
            )

        if not isinstance(extended_set_weight, float):
            raise TypeError(
                'Expected float for extended_set_weight'
                f', got {type(extended_set_weight)}',
            )

        if decay_reset_interval < 1:
            raise ValueError('Decay reset interval must be a positive integer.')

        if extended_set_size < 0:
            raise ValueError('Extended set size must be a nonnegative integer.')

        self.decay_delta = decay_delta
        self.decay_reset_interval = decay_reset_interval
        self.decay_reset_on_gate = decay_reset_on_gate
        self.extended_set_size = extended_set_size
        self.extended_set_weight = extended_set_weight

    def forward_pass(
        self,
        circuit: Circuit,
        pi: list[int],
        cg: CouplingGraph,
        modify_circuit: bool = False,
    ) -> None:
        """
        Apply a forward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                maps logical qudits to physical qudits. So, `pi[l] == p`
                implies logical qudit `l` is sitting on physical qudit `p`.

            cg (CouplingGraph): The connectivity of the hardware.

            modfiy_circuit (bool): Whether to modify the circuit as the
                pass is applied or not. (Default: False)
        """
        # Preprocessing
        D = cg.all_pairs_shortest_path()
        F = circuit.front
        decay = [1.0 for i in range(circuit.num_qudits)]
        iter_count = 0
        prev_swap = (-1, -1)
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        _logger.debug(f'Starting forward sabre pass with pi: {pi}.')

        if modify_circuit:
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        # Main Loop
        while len(F) > 0:

            # Retrieve executable gates giving the current mapping `pi`
            execute_list = [n for n in F if self._can_exe(circuit[n], pi, cg)]

            # Execute the gates and update F
            if len(execute_list) > 0:
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')

                    if modify_circuit:
                        op = circuit[n]
                        physical_location = [pi[q] for q in op.location]
                        mapped_circuit.append_gate(
                            op.gate,
                            physical_location,
                            op.params,
                        )

                    # Reset previous swap if executed gate overlaps it
                    if any(pi[i] in prev_swap for i in circuit[n].location):
                        prev_swap = (-1, -1)

                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)

                # Reset decay if necessary
                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0

                continue  # Restart main loop if we executed at least one gate

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(
                circuit, F, E, D, cg, pi, decay, prev_swap,
            )
            self._apply_swap(best_swap, pi, decay)
            prev_swap = best_swap
            if modify_circuit:
                mapped_circuit.append_gate(SwapGate(), best_swap)

            # Update loop counter and reset decay if necessary
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

        if modify_circuit:
            circuit.become(mapped_circuit)

    def backward_pass(
        self,
        circuit: Circuit,
        pi: list[int],
        cg: CouplingGraph,
    ) -> None:
        """
        Apply a backward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                maps logical qudits to physical qudits. So, `pi[l] == p`
                implies logical qudit `l` is sitting on physical qudit `p`.

            cg (CouplingGraph): The connectivity of the hardware.
        """
        # Preprocessing
        D = cg.all_pairs_shortest_path()
        F = circuit.rear
        decay = [1.0 for i in range(circuit.num_qudits)]
        iter_count = 0
        prev_swap = (-1, -1)
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        _logger.debug(f'Starting backward sabre pass with pi: {pi}.')

        # Main Loop
        while len(F) > 0:

            # Retrieve executable gates giving the current mapping: pi
            execute_list = [n for n in F if self._can_exe(circuit[n], pi, cg)]

            # Execute the gates and update F
            if len(execute_list) > 0:
                for n in execute_list:
                    F.remove(n)
                    next_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')

                    # Reset previous swap if executed gate overlaps it
                    if any(pi[i] in prev_swap for i in circuit[n].location):
                        prev_swap = (-1, -1)

                    for predessor in circuit.prev(n):
                        if predessor not in next_executed_counts:
                            next_executed_counts[predessor] = 1
                        else:
                            next_executed_counts[predessor] += 1
                        num_next_executed = next_executed_counts[predessor]
                        total_num_next = len(circuit.next(predessor))
                        if num_next_executed == total_num_next:
                            F.add(predessor)

                # Reset decay if necessary
                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0

                continue  # Restart main loop if we executed at least one gate

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(
                circuit, F, E, D, cg, pi, decay, prev_swap,
            )
            self._apply_swap(best_swap, pi, decay)
            prev_swap = best_swap

            # Update loop counter and reset decay if necessary
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

    def _can_exe(self, op: Operation, pi: list[int], cg: CouplingGraph) -> bool:
        """Return true if `op` is executable given the current mapping `pi`."""
        # TODO: check if circuitgate of only 1-qubit gates
        if op.num_qudits == 1:
            return True
        physical_qudits = [pi[i] for i in op.location]
        return cg.get_subgraph(physical_qudits).is_fully_connected()

    def _calc_extended_set(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
    ) -> set[CircuitPoint]:
        """Calculate the Extended Set for look-ahead capabilities."""
        extended_set: set[CircuitPoint] = set()
        frontier = list(copy.copy(F))
        while len(frontier) > 0 and len(extended_set) < self.extended_set_size:
            n = frontier.pop(0)
            extended_set.update(circuit.next(n))
            frontier.extend(circuit.next(n))
        return extended_set

    def _get_best_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        E: set[CircuitPoint],
        D: list[list[int]],
        cg: CouplingGraph,
        pi: list[int],
        decay: list[float],
        prev_swap: tuple[int, int],
    ) -> tuple[int, int]:
        """Return the best swap given the current algorithm state."""
        # Track best one
        best_score = np.inf
        best_swap = None

        # Gather all considerable swaps
        swap_candidate_list = self._obtain_swaps(circuit, F, pi, cg)
        if prev_swap in swap_candidate_list:
            swap_candidate_list.remove(prev_swap)

        # Score them, tracking the best one
        for swap in swap_candidate_list:
            score = self._score_swap(circuit, F, pi, D, swap, decay, E)
            if score < best_score:
                best_score = score
                best_swap = swap

        if best_swap is None:
            raise RuntimeError('Unable to find best swap.')

        return best_swap

    def _obtain_swaps(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pi: list[int],
        cg: CouplingGraph,
    ) -> set[tuple[int, int]]:
        """Produce all physical swaps with at least one qudit in F."""
        all_qudits: set[int] = set()
        for n in F:
            all_qudits.update(circuit[n].location)
        physical_qudits = [pi[i] for i in all_qudits]

        swaps = set()
        for physical_qudit in physical_qudits:
            neighbors = cg.get_neighbors_of(physical_qudit)
            for neighbor in neighbors:
                a = min(neighbor, physical_qudit)
                b = max(neighbor, physical_qudit)
                swaps.add((a, b))

        return swaps

    def _score_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pi: list[int],
        D: list[list[int]],
        swap: tuple[int, int],
        decay: list[float],
        E: set[CircuitPoint],
    ) -> float:
        """Score the candidate swap given the current algorithm state."""
        # Apply potential swap
        l1, l2 = pi.index(swap[0]), pi.index(swap[1])
        pi[l1], pi[l2] = pi[l2], pi[l1]

        # Calculate front set term
        front = 0.0
        for n in F:
            logical_qudits = circuit[n].location

            # Disallow meaningless swaps
            physical_qudits = [pi[i] for i in logical_qudits]
            if swap[0] in physical_qudits and swap[1] in physical_qudits:
                pi[l1], pi[l2] = pi[l2], pi[l1]
                return np.inf

            min_term = np.inf
            for q in logical_qudits:
                term = 0.0
                for p in logical_qudits:
                    if p == q:
                        continue
                    term += D[pi[q]][pi[p]]
                min_term = min(term, min_term)
            front += min_term
        front /= len(F)

        # Calculate extended set term
        extend = 0.0
        if len(E) > 0:
            for n in E:
                logical_qudits = circuit[n].location
                min_term = np.inf
                for q in logical_qudits:
                    term = 0.0
                    for p in logical_qudits:
                        if p == q:
                            continue
                        term += D[pi[q]][pi[p]]
                    min_term = min(term, min_term)
                extend += min_term
            extend /= len(E)
            extend *= self.extended_set_weight

        # Calculate decay factor
        decay_factor = max(decay[swap[0]], decay[swap[1]])

        # Undo potential swap
        pi[l1], pi[l2] = pi[l2], pi[l1]

        # Return final score
        return decay_factor * (front + extend)

    def _apply_swap(
        self,
        swap: tuple[int, int],
        pi: list[int],
        decay: list[float],
    ) -> None:
        """Apply the swap to `pi` and update `decay`."""
        _logger.debug('applying swap %s' % str(swap))
        l1, l2 = pi.index(swap[0]), pi.index(swap[1])
        pi[l1], pi[l2] = pi[l2], pi[l1]

        decay[swap[0]] += self.decay_delta
        decay[swap[1]] += self.decay_delta
