"""Search based synthesis #TODO """


from typing import Any
from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.compiler.passes.synthesispass import SynthesisPass


class SearchBasedReallyLongPotentiallyRedoneNameSynthesisPass(SynthesisPass):

    def __init__(
        self,
        heuristic_function: HeuristicFunction,
        layer_generator: LayerGenerator,
    ) -> None:
        self.heuristic_function = heuristic_function
        self.frontier = PriorityQueue  # TODO: thread-safe priorityqueue with hash updating
    
    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry` into a circuit, see SynthesisPass for more info."""
        pass


class AugmentedSearchPass(SearchBasedReallyLongPotentiallyRedoneNameSynthesisPass):

    def layer_generator(...):
        # re write it here and done