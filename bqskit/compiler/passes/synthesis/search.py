"""Search based synthesis #TODO."""
from __future__ import annotations

from typing import Any

from bqskit.compiler.passes.synthesispass import SynthesisPass
from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary import UnitaryMatrix


class SearchBasedReallyLongPotentiallyRedoneNameSynthesisPass(SynthesisPass):

    def __init__(
        self,
        # heuristic_function: HeuristicFunction,
        # layer_generator: LayerGenerator,  # TODO:
    ) -> None:
        # self.heuristic_function = heuristic_function
        # TODO: thread-safe priorityqueue with hash updating
        # self.frontier = PriorityQueue
        pass  # TODO

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry` into a circuit, see SynthesisPass for more info."""
        pass  # TODO
