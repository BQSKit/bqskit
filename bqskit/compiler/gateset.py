"""This module implements the GateSet class."""
from __future__ import annotations

import sys
from collections.abc import Set
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constant.csum import CSUMGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.parameterized.cphase import ArbitraryCPhaseGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.u8 import U8Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence_of_int

if TYPE_CHECKING:
    from bqskit.passes.search.generator import LayerGenerator

if sys.version_info < (3, 9):
    SuperType = Set
else:
    SuperType = Set[Gate]


class GateSet(SuperType):
    """A set of a quantum processing unit's supported quantum gates."""

    def __init__(self, gates: GateSetLike) -> None:
        """Initialize a GateSet object."""
        if isinstance(gates, GateSet):
            self._gates: frozenset[Gate] = gates._gates.copy()
            self.radix_set: frozenset[int] = gates.radix_set.copy()
            return

        if isinstance(gates, Gate):
            gates = [gates]

        radix_set: set[int] = set()
        gate_set: set[Gate] = set()
        for g in gates:
            radix_set.update(g.radixes)
            gate_set.add(g)
        self.radix_set = frozenset(radix_set)
        self._gates = frozenset(gate_set)

    def build_layer_generator(self) -> LayerGenerator:
        """Build a standard layer generator compliant with this gate set."""
        if not any(isinstance(g, GeneralGate) for g in self.single_qudit_gates):
            raise RuntimeError(
                'Cannot automatically build a layer generator for a gate set'
                " without any general single-qubit gates. Instead, it's best"
                ' to use `build_mq_layer_generator` and then do single-qudit'
                ' gate retargeting afterwards.',
            )

        from bqskit.passes.search.generators import SimpleLayerGenerator
        from bqskit.passes.search.generators import WideLayerGenerator

        sqs = [g for g in self.single_qudit_gates if isinstance(g, GeneralGate)]
        sq = sqs[0]

        if len(self.two_qudit_gates) == 1 and len(self.many_qudit_gates) == 0:
            gate = list(self.two_qudit_gates)[0]
            return SimpleLayerGenerator(gate, sq)

        return WideLayerGenerator(self.multi_qudit_gates, sq)

    def build_mq_layer_generator(self) -> LayerGenerator:
        """
        Produce a layer generator using this gate set's multi-qudit gates.

        The resulting layer generator will use arbitrary single-qudit gates.
        """
        from bqskit.passes.search.generators import FourParamGenerator
        from bqskit.passes.search.generators import SimpleLayerGenerator
        from bqskit.passes.search.generators import WideLayerGenerator

        if len(self.two_qudit_gates) == 1 and len(self.many_qudit_gates) == 0:
            if CNOTGate() in self:
                return FourParamGenerator()

            gate = list(self.two_qudit_gates)[0]

            if isinstance(gate, GeneralGate):
                return WideLayerGenerator(gate, None)

            return SimpleLayerGenerator(gate, self.get_general_sq_gate())

        return WideLayerGenerator(
            self.multi_qudit_gates,
            self.get_general_sq_gate(),
        )

    def get_general_sq_gate(self) -> GeneralGate:
        """
        Return the gate set's arbitrary single-qudit gate.

        During off-the-shelf BQSKit compilations, single-qudit gate rebasing is
        done last. The earlier stages of compilation focus on multi-qudit gates
        and use arbitrary single-qudit rotations. This function will return the
        best arbitrary single-qudit rotation to use during these steps based on
        the gates in the set.
        """
        if len(self.radix_set) == 0:
            raise RuntimeError('Cannot suggest single-qudit gate for empty set')

        if len(self.radix_set) != 1:
            raise RuntimeError(
                'Cannot automatically suggest general single-qudit gate for'
                ' mixed radix systems.',
            )

        if U3Gate() in self:
            return U3Gate()

        if U8Gate() in self:
            return U8Gate()

        for g in self:
            if isinstance(g, VariableUnitaryGate) and g.num_qudits == 1:
                return g

        for g in self:
            if isinstance(g, GeneralGate) and g.num_qudits == 1:
                return g

        radix = list(self.radix_set)[0]

        if radix == 2:
            return U3Gate()

        return VariableUnitaryGate(1, [radix])

    @property
    def single_qudit_gates(self) -> set[Gate]:
        """All gates that act on only one qudit."""
        return {g for g in self if g.num_qudits == 1}

    @property
    def two_qudit_gates(self) -> set[Gate]:
        """All gates that act on two qudits."""
        return {g for g in self if g.num_qudits == 2}

    @property
    def many_qudit_gates(self) -> set[Gate]:
        """All gates that act on more than two qudits."""
        return {g for g in self if g.num_qudits > 2}

    @property
    def multi_qudit_gates(self) -> set[Gate]:
        """All gates that act on more than one qudits."""
        return {g for g in self if g.num_qudits >= 2}

    @staticmethod
    def default_gate_set(radixes: Sequence[int] | int = 2) -> GateSet:
        """Build a default gate set for a given `radixes`."""
        if is_integer(radixes):
            radixes = [radixes]

        assert is_sequence_of_int(radixes)

        if len(radixes) == 0 or all(r == 2 for r in radixes):
            return GateSet({CNOTGate(), U3Gate()})

        unique_radixes = set(radixes)

        gates: set[Gate] = set()
        for r in unique_radixes:
            gates.add(VariableUnitaryGate(1, [r]))

        if len(unique_radixes) == 1:
            gates.add(CSUMGate(list(unique_radixes)[0]))
            return GateSet(gates)

        for r1 in unique_radixes:
            for r2 in unique_radixes:
                if r1 <= r2:
                    gates.add(ArbitraryCPhaseGate([r1, r2]))

        return GateSet(gates)

    def __iter__(self) -> Iterator[Gate]:
        """Iterator for this gate set's gates."""
        return self._gates.__iter__()

    def __len__(self) -> int:
        """Number of gates in the set."""
        return len(self._gates)

    def __contains__(self, obj: object) -> bool:
        """Return true if a gate is in the set."""
        return self._gates.__contains__(obj)

    def union(self, *others: Any) -> GateSet:
        """Return a new GateSet with elements from this one and all others."""
        return GateSet(self._gates.union(*others))

    def intersection(self, *others: Any) -> GateSet:
        """Return a new GateSet with elements common to all sets."""
        return GateSet(self._gates.intersection(*others))

    def difference(self, *others: Any) -> GateSet:
        """Return a new GateSet with this set's elements but not the others."""
        return GateSet(self._gates.difference(*others))

    def symmetric_difference(self, other: Iterable[Gate]) -> GateSet:
        """Return a new GateSet with elements in either set but not both."""
        return GateSet(self._gates.symmetric_difference(other))

    def issubset(self, other: Iterable[Gate]) -> bool:
        """Test whether every element in the gate set is in other."""
        return self._gates.issubset(other)

    def issuperset(self, other: Iterable[Gate]) -> bool:
        """Report whether this gate set contains `other`."""
        return self._gates.issuperset(other)

    def __str__(self) -> str:
        """String representation of the GateSet."""
        return self._gates.__str__().replace('frozenset', 'GateSet')

    def __repr__(self) -> str:
        """Detailed representation of the GateSet."""
        return self._gates.__repr__().replace('frozenset', 'GateSet')
    
    def __hash__(self) -> int:
        """Hash of the GateSet."""
        return self.__repr__().__hash__()


GateSetLike = Union[GateSet, Iterable[Gate], Gate]
