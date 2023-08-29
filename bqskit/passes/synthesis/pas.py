"""This module implements the PermutationAwareSynthesisPass pass."""
from __future__ import annotations

import itertools as it
import logging
from typing import Callable

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.permutation import PermutationMatrix
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.runtime import get_runtime


_logger = logging.getLogger(__name__)


def op_count(circuit: Circuit) -> float:
    """Counts the number of operations in a circuit."""
    return float(circuit.num_operations)


class PermutationAwareSynthesisPass(SynthesisPass):
    """Synthesis pass considering target permutations."""

    def __init__(
        self,
        input_perm: bool = False,
        output_perm: bool = True,
        inner_synthesis: SynthesisPass = LEAPSynthesisPass(),
        scoring_fn: Callable[[Circuit], float] = op_count,
    ) -> None:
        """
        Construct a PermutationAwareSynthesisPass.

        Args:
            input_perm (bool): If true, vary the input permutation
                during synthesis. (Default: False)

            output_perm (bool): If true, vary the output permutation
                during synthesis. (Default: True)

            inner_synthesis (SynthesisPass): The synthesis algorithm used
                on all permutations. (Default: :class:`LEAPSynthesisPass`)

            scoring_fn (Callable[[Circuit], float]): The function used to
                score the permuted circuits. The smallest score wins.
                (Default: :func:`op_count`)
        """
        # TODO: Add option to choose first returned result
        if not isinstance(inner_synthesis, SynthesisPass):
            bad_type = type(inner_synthesis)
            raise TypeError(f'Expected SynthesisPass object, got {bad_type}.')

        if not callable(scoring_fn):
            bad_type = type(scoring_fn)
            m = f'Expected a function from circuits to scores, got {bad_type}.'
            raise TypeError(m)

        self.inner_synthesis = inner_synthesis
        self.scoring_fn = scoring_fn
        self.input_perm = input_perm
        self.output_perm = output_perm

    async def synthesize(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        # Calculate all permuted targets
        width = utry.num_qudits
        perms = list(it.permutations(range(width)))
        no_perm = [tuple(range(width))]
        Pis = [PermutationMatrix.from_qubit_location(width, p) for p in perms]
        Pos = [PermutationMatrix.from_qubit_location(width, p) for p in perms]

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

        # Synthesize all permuted targets
        circuits: list[Circuit] = await get_runtime().map(
            self.inner_synthesis.synthesize,
            targets,
            [data] * len(targets),
        )

        # Return best circuit
        best_circuit = circuits[0]
        best_perm = permsbyperms[0]
        best_score = self.scoring_fn(circuits[0])
        for perm, circuit in zip(permsbyperms[1:], circuits[1:]):
            score = self.scoring_fn(circuit)
            if score < best_score:
                best_circuit = circuit
                best_perm = perm
                best_score = score

        data['initial_mapping'] = best_perm[0]
        data['final_mapping'] = best_perm[1]
        return best_circuit
