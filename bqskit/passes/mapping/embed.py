"""This module implements the EmbedAllPermutationsPass pass."""
from __future__ import annotations

import copy
import itertools as it
import logging
from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.mapping.topology import SubtopologySelectionPass
from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.permutation import PermutationMatrix
from bqskit.runtime import get_runtime


_logger = logging.getLogger(__name__)


def multi_qudit_op_count(circuit: Circuit) -> float:
    """Counts the number of multi-qudit operations in a circuit."""
    x = sum(c for g, c in circuit.gate_counts.items() if g.num_qudits >= 2)
    return float(x)


class EmbedAllPermutationsPass(BasePass):
    """Embed permutation aware synthesis results into a flow for future use."""

    def __init__(
        self,
        input_perm: bool = False,
        output_perm: bool = True,
        vary_topology: bool = True,
        inner_synthesis: SynthesisPass = LEAPSynthesisPass(),
        scoring_fn: Callable[[Circuit], float] = multi_qudit_op_count,
    ) -> None:
        """
        Construct a EmbedAllPermutationsPass.

        Args:
            input_perm (bool): If true, vary the input permutation
                during synthesis. (Default: False)

            output_perm (bool): If true, vary the output permutation
                during synthesis. (Default: True)

            vary_topology (bool): If true, vary the desired coupling graph
                during synthesis. (Default: True)

            inner_synthesis (SynthesisPass): The synthesis algorithm used
                on all permutations. (Default: :class:`LEAPSynthesisPass`)

            scoring_fn (Callable[[Circuit], float]): The scoring function
                used when comparing the original circuit to the synthesized
                circuit with the same configuration. The smallest score wins.
                (Default: :func:`multi_qudit_op_count`)
        """
        if not isinstance(inner_synthesis, SynthesisPass):
            bad_type = type(inner_synthesis)
            raise TypeError(f'Expected SynthesisPass object, got {bad_type}.')

        if not callable(scoring_fn):
            bad_type = type(scoring_fn)
            m = f'Expected a function from circuits to scores, got {bad_type}.'
            raise TypeError(m)

        self.input_perm = input_perm
        self.output_perm = output_perm
        self.vary_topology = vary_topology
        self.inner_synthesis = inner_synthesis
        self.scoring_fn = scoring_fn

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        utry = data.target

        if not all(r == utry.radixes[0] for r in utry.radixes):
            raise NotImplementedError(
                'PermutationAwareSynthesisPass only supports unitaries '
                'with the same radix on all qudits currently.',
            )

        # Calculate all permuted targets
        width = utry.num_qudits
        perms = list(it.permutations(range(width)))
        no_perm = [tuple(range(width))]
        Pis = [
            PermutationMatrix.from_qudit_location(width, utry.radixes[0], p)
            for p in perms
        ]
        Pos = [
            PermutationMatrix.from_qudit_location(width, utry.radixes[0], p)
            for p in perms
        ]

        if self.input_perm and self.output_perm:
            permsbyperms = list(it.product(perms, perms))
            targets = [Po.T @ utry @ Pi for Pi, Po in it.product(Pis, Pos)]

        elif self.input_perm:
            permsbyperms = list(it.product(perms, no_perm))
            targets = [utry @ Pi for Pi in Pis]

        elif self.output_perm:
            permsbyperms = list(it.product(no_perm, perms))
            targets = [Po.T @ utry for Po in Pos]

        else:
            _logger.warning('No permutation is being used in PAS.')
            permsbyperms = list(it.product(no_perm, no_perm))
            targets = [utry]

        # Calculate all target coupling graphs
        if self.vary_topology and width != 1:
            if SubtopologySelectionPass.key not in data:
                raise RuntimeError(
                    'Cannot find subtopologies, try running a'
                    ' SubtopologySelectionPass first.',
                )

            if width not in data[SubtopologySelectionPass.key]:
                raise RuntimeError(
                    'Subtopology information for block size'
                    f' {width} is not available.',
                )

            graphs = data[SubtopologySelectionPass.key][width]

        else:
            graphs = [CouplingGraph.all_to_all(width)]

        datas = []
        for graph in graphs:
            model = MachineModel(
                circuit.num_qudits, graph,
                data.gate_set, data.model.radixes,
            )
            target_data = copy.deepcopy(data)
            target_data.model = model
            datas.append(target_data)

        # Create parallel arrays for map
        extended_targets = []
        extended_datas = []
        for t, d in it.product(targets, datas):
            extended_targets.append(t)
            extended_datas.append(d)

        # Synthesize all permuted targets
        circuits: list[Circuit] = await get_runtime().map(
            self.inner_synthesis.synthesize,
            extended_targets,
            extended_datas,
        )

        # Store results
        perm_data: dict[
            CouplingGraph,
            dict[tuple[tuple[int, ...], tuple[int, ...]], Circuit],
        ] = {}
        for i, c in enumerate(circuits):
            graph = graphs[i % len(graphs)]
            perm = permsbyperms[i // len(graphs)]
            if graph not in perm_data:
                perm_data[graph] = {}

            if perm in perm_data[graph]:
                # Update if it is better than whats already there
                s1 = self.scoring_fn(perm_data[graph][perm])
                s2 = self.scoring_fn(c)
                if s2 < s1:
                    perm_data[graph][perm] = c
            else:
                perm_data[graph][perm] = c

            # Calculate number of multi-qudit gates
            num_mq_gates = 0
            for gate, count in c.gate_counts.items():
                if gate.num_qudits >= 2:
                    num_mq_gates += count

            # Generate the extra circuits through universal permutations
            all_perms = list(it.permutations(range(width)))
            for univ_perm in all_perms[1:]:
                renumber_c = c.copy()
                renumber_c.renumber_qudits(univ_perm)
                new_pi = tuple(univ_perm[i] for i in perm[0])
                new_pf = tuple(univ_perm[i] for i in perm[1])
                new_graph = renumber_c.coupling_graph
                if new_graph not in perm_data:
                    perm_data[new_graph] = {}

                new_perm = (new_pi, new_pf)
                if new_perm not in perm_data[new_graph]:
                    perm_data[new_graph][new_perm] = renumber_c
                else:
                    s1 = self.scoring_fn(perm_data[new_graph][new_perm])
                    s2 = self.scoring_fn(renumber_c)
                    if s2 < s1:
                        perm_data[new_graph][new_perm] = renumber_c

        # Override no perm result if original is better and compatible
        if circuit.gate_set.issubset(data.model.gate_set):
            for univ_perm in it.permutations(range(width)):
                # Permute original circuit and override worse results
                uperm = (univ_perm, univ_perm)
                renumber_c = circuit.copy()
                renumber_c.renumber_qudits(univ_perm)
                new_graph = renumber_c.coupling_graph
                new_score = self.scoring_fn(renumber_c)
                for graph, graph_data in perm_data.items():
                    if all(e in graph for e in new_graph):
                        if uperm not in graph_data:
                            graph_data[uperm] = renumber_c
                        else:
                            if new_score < self.scoring_fn(graph_data[uperm]):
                                graph_data[uperm] = renumber_c

        # Record permutation data in the pass data
        data['permutation_data'] = perm_data
