"""This module implements the PAMRoutingPass class."""
from __future__ import annotations

import itertools as it
import logging
from typing import Dict
from typing import Sequence
from typing import Tuple

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import SwapGate
from bqskit.ir.point import CircuitPoint
from bqskit.passes.mapping.sabre import GeneralizedSabreAlgorithm
from bqskit.qis.graph import CouplingGraph

_logger = logging.getLogger(__name__)


PAMBlockPermData = Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Circuit]
PAMBlockTAPermData = Dict[CouplingGraph, PAMBlockPermData]


class PermutationAwareMappingAlgorithm(GeneralizedSabreAlgorithm):
    """Route the circuit with permutation awareness."""

    def __init__(
        self,
        gate_count_weight: float = 0.1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
    ) -> None:
        """
        Construct a PermutationAwareMappingAlgorithm.

        Args:
            gate_count_weight (float): The weight on block gate count
                versus mapping score when selecting a permutation.

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
        if not isinstance(gate_count_weight, float):
            bad_type = type(gate_count_weight)
            m = f'Expected float for gate_count_weight, got {bad_type}'
            raise TypeError(m)

        self.gate_count_weight = gate_count_weight

        super().__init__(
            decay_delta,
            decay_reset_interval,
            decay_reset_on_gate,
            extended_set_size,
            extended_set_weight,
        )

    def forward_pass(  # type: ignore
        self,
        circuit: Circuit,
        pi: list[int],
        cg: CouplingGraph,
        perm_data: dict[CircuitPoint, PAMBlockTAPermData],
        modify_circuit: bool = False,
    ) -> None:
        """
        Apply a forward pass of the PAM algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                maps logical qudits to physical qudits. So, `pi[l] == p`
                implies logical qudit `l` is sitting on physical qudit `p`.

            cg (CouplingGraph): The connectivity of the hardware.

            perm_data (dict[CircuitPoint, PAMBlockTAPermData]):
                Maps each permutation configuration for every block.

            modfiy_circuit (bool): Whether to modify the circuit as the
                pass is applied or not. (Default: False)
        """
        # Preprocessing
        D = cg.all_pairs_shortest_path()
        F = circuit.front
        decay = [1.0 for i in range(circuit.num_qudits)]
        iter_count = 0
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_swaps: list[tuple[int, int]] = []
        _logger.debug(f'Starting forward pam pass with pi: {pi}.')

        if modify_circuit:
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        # Main Loop
        while len(F) > 0:

            # Retrieve executable gates giving the current mapping `pi`
            execute_list = [n for n in F if self._can_exe(circuit[n], pi, cg)]

            # Execute the gates and update F
            if len(execute_list) > 0:
                leading_swaps = []

                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')

                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)

                # Permute the qubits on the just executed gates
                E = self._calc_extended_set(circuit, F)
                for n in execute_list:
                    op = circuit[n]
                    p1, circ, p2 = self._get_best_perm(
                        circuit,
                        perm_data[n],
                        cg,
                        F,
                        pi,
                        D,
                        E,
                        op.location,
                    )

                    self._apply_perm(p1, pi)

                    if modify_circuit:
                        physical_location = [pi[q] for q in op.location]
                        mapped_circuit.append_circuit(
                            circ,
                            physical_location,
                            True,
                        )

                    self._apply_perm(p2, pi)

                # Reset decay if necessary
                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0

                continue  # Restart main loop if we executed at least one gate

            # If execute list is empty, check for local-minima
            elif len(leading_swaps) > 5 * cg.num_qudits:
                _logger.debug('Sabre stuck in local minima, backtracking...')

                # Backtrack by removing leading swaps
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pi, decay)
                    if modify_circuit:
                        point = mapped_circuit._rear[swap[0]]
                        mapped_circuit.pop(point)
                leading_swaps = []

                # Override heuristic search to progress
                _logger.debug('Overriding sabre search...')
                all_logical_qudits = [circuit[n].location for n in F]
                qudits = min(
                    all_logical_qudits,
                    key=lambda qs: self._get_distance(qs, pi, D),
                )
                for swap in self._uphill_swaps(qudits, cg, pi, D):
                    self._apply_swap(swap, pi, decay)
                    if modify_circuit:
                        mapped_circuit.append_gate(SwapGate(), swap)
                _logger.debug('Stopping override.')
                continue

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(circuit, F, E, D, cg, pi, decay)
            self._apply_swap(best_swap, pi, decay)
            leading_swaps.append(best_swap)

            if modify_circuit:
                mapped_circuit.append_gate(SwapGate(), best_swap)

            # Update loop counter and reset decay if necessary
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

        if modify_circuit:
            circuit.become(mapped_circuit)

    def _get_best_perm(
        self,
        circuit: Circuit,
        perm_data: PAMBlockTAPermData,
        cg: CouplingGraph,
        F: set[CircuitPoint],
        pi: list[int],
        D: list[list[int]],
        E: set[CircuitPoint],
        qudits: Sequence[int],
    ) -> tuple[tuple[int, ...], Circuit, tuple[int, ...]]:
        """Return the best permutation to apply before and after a gate."""

        # Local permutations determine how a gate is permuted in it own space
        local_perms = list(it.permutations(range(len(qudits))))

        # Global perms capture local perms' effect on the full logical space
        global_perms = [
            tuple(qudits[i] for i in lperm)
            for lperm in local_perms
        ]

        # Inverted Permutations
        inv_local_perms = [
            tuple(lperm.index(i) for i in range(len(qudits)))
            for lperm in local_perms
        ]

        # Inverted Global Permutations
        inv_global_perms = [
            tuple(qudits[i] for i in ilperm)
            for ilperm in inv_local_perms
        ]

        # Gather valid pre, circ, post triples
        pre_circ_post_triples = []
        perm_iter = zip(local_perms, inv_local_perms, inv_global_perms)
        for lperm, ilperm, gperm1 in perm_iter:
            physical_location = [pi[qudits[p]] for p in ilperm]
            local_graph = cg.get_subgraph(physical_location)
            if local_graph in perm_data:
                for perms, circ in perm_data[local_graph].items():
                    if lperm == perms[0]:
                        gperm2 = global_perms[local_perms.index(perms[1])]
                        pre_circ_post_triples.append((gperm1, circ, gperm2))

        if len(pre_circ_post_triples) == 0:
            raise RuntimeError(
                'Unable to find any valid permutated circuits.\n'
                'You must embed proper permutation aware synthesis results'
                ' first before running this pass.\n'
                'If you are already running an'
                ' EmbedAllPermutationsPass, try toggling topology selection.',
            )

        # For each permutation get the entangling gate count
        mq_gate_counts = []
        sq_gate_counts = []
        for _, circ, _ in pre_circ_post_triples:
            num_tq_gates = 0
            num_sq_gates = 0
            for gate, count in circ.gate_counts.items():
                if gate.num_qudits >= 2:
                    num_tq_gates += count
                else:
                    num_sq_gates += count
            mq_gate_counts.append(num_tq_gates)
            sq_gate_counts.append(num_sq_gates)

        # If no more gates after this one, then pick the shortest circuit
        if len(F) == 0:
            best_index = np.argmin(mq_gate_counts)
            return pre_circ_post_triples[best_index]

        # Calculate best scoring permutation
        best_triple = pre_circ_post_triples[0]
        best_perm = (best_triple[0], best_triple[2])
        mapping_score = self._score_perm(circuit, F, pi, D, best_perm, E)
        gate_score = mq_gate_counts[0] * self.gate_count_weight / len(F)
        best_score = mapping_score + gate_score

        for i in range(1, len(pre_circ_post_triples)):
            gperm = (pre_circ_post_triples[i][0], pre_circ_post_triples[i][2])
            score = self._score_perm(circuit, F, pi, D, gperm, E)
            score = mq_gate_counts[i] * self.gate_count_weight / len(F) + score
            if score < best_score:
                best_score = score
                best_perm = gperm
                best_triple = pre_circ_post_triples[i]
        return best_triple

    def _score_perm(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pi: list[int],
        D: list[list[int]],
        perm: tuple[Sequence[int], Sequence[int]],
        E: set[CircuitPoint],
    ) -> float:
        """Calculating the routing score after applying `perm`."""
        pi_bkp = pi.copy()
        pi_c = {q: pi[perm[0][i]] for i, q in enumerate(sorted(perm[0]))}
        for q in perm[0]:
            pi[q] = pi_c[q]
        pi_c = {q: pi[perm[1][i]] for i, q in enumerate(sorted(perm[1]))}
        for q in perm[1]:
            pi[q] = pi_c[q]

        # Front Set Term
        front = 0.0
        for n in F:
            min_term = np.inf
            for q in circuit[n].location:
                term = 0.0
                for p in circuit[n].location:
                    if p == q:
                        continue
                    term += D[pi[q]][pi[p]]
                min_term = min(term, min_term)
            front += min_term
        front /= len(F)

        # Extended Set Term
        extend = 0.0
        if len(E) > 0:
            for n in E:
                min_term = np.inf
                for q in circuit[n].location:
                    term = 0.0
                    for p in circuit[n].location:
                        if p == q:
                            continue
                        term += D[pi[q]][pi[p]]
                    min_term = min(term, min_term)
                extend += min_term
            extend /= len(E)
            extend *= self.extended_set_weight

        pi[:] = pi_bkp[:]
        return front + extend

    def _apply_perm(self, perm: Sequence[int], pi: list[int]) -> None:
        """Apply the `perm` permutation to the current mapping `pi`."""
        _logger.debug('applying permutation %s' % str(perm))
        pi_c = {q: pi[perm[i]] for i, q in enumerate(sorted(perm))}
        for q in perm:
            pi[q] = pi_c[q]
