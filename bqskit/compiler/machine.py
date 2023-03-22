"""This module implements the MachineModel class."""
from __future__ import annotations

from typing import Sequence
from typing import TYPE_CHECKING

from bqskit.ir.gate import Gate
from bqskit.ir.gates.constant.csum import CSUMGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.u8 import U8Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.location import CircuitLocation
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.graph import CouplingGraphLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_valid_radixes

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


default_qubit_gate_set: set[Gate] = {
    CNOTGate(),
    U3Gate(),
}
default_qutrit_gate_set: set[Gate] = {
    CSUMGate(),
    U8Gate(),
}


class MachineModel:
    """A model of a quantum processing unit."""

    def __init__(
        self,
        num_qudits: int,
        coupling_graph: CouplingGraphLike | None = None,
        gate_set: set[Gate] | None = None,
        radixes: Sequence[int] = [],
    ) -> None:
        """
        MachineModel Constructor.

        Args:
            num_qudits (int): The total number of qudits in the machine.

            coupling_graph (Iterable[tuple[int, int]] | None): A coupling
                graph describing which pairs of qudits can interact.
                Given as an undirected edge set. If left as None, then
                an all-to-all coupling graph is used as a default.
                (Default: None)

            gate_set (set[Gate]): The native gate set available on the
                machine. Defaults to CNOTs+U3s for qubits, CSUM+U8s for
                qutrits, and variable unitary gates for arbitrary qudits.

            radixes (Sequence[int]): A sequence with its length equal
                to `num_qudits`. Each element specifies the base of a
                qudit. Defaults to qubits.

        Raises:
            ValueError: If `num_qudits` is nonpositive.

        Note:
            Pre-built models for many active QPUs exist in the `bqskit.ext`
            package.
        """

        if not is_integer(num_qudits):
            raise TypeError(
                f'Expected integer num_qudits, got {type(num_qudits)}.',
            )

        if num_qudits <= 0:
            raise ValueError(f'Expected positive num_qudits, got {num_qudits}.')

        self.radixes = tuple(radixes if len(radixes) > 0 else [2] * num_qudits)

        if not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if len(self.radixes) != num_qudits:
            raise ValueError(
                'Expected length of radixes to be equal to num_qudits:'
                ' %d != %d' % (len(self.radixes), num_qudits),
            )

        if coupling_graph is None:
            coupling_graph = CouplingGraph.all_to_all(num_qudits)

        if not CouplingGraph.is_valid_coupling_graph(
                coupling_graph, num_qudits,
        ):
            raise TypeError('Invalid coupling graph, expected list of tuples')

        if gate_set is None:
            r = self.radixes[0]
            if r == 2:
                gate_set = default_qubit_gate_set
            elif r == 3:
                gate_set = default_qutrit_gate_set
            else:
                gate_set = {VariableUnitaryGate(2, [r, r])}

        if not isinstance(gate_set, set):
            raise TypeError(f'Expected set of gates, got {type(gate_set)}.')

        if not all(isinstance(g, Gate) for g in gate_set):
            raise TypeError(f'Expected set of gates, got {type(gate_set)}.')

        self.gate_set = gate_set
        self.coupling_graph = CouplingGraph(coupling_graph)
        self.num_qudits = num_qudits

    def get_locations(self, block_size: int) -> list[CircuitLocation]:
        """Return all `block_size` connected blocks of qudit indicies."""
        return self.coupling_graph.get_subgraphs_of_size(block_size)

    # TODO: add placement support
    def is_compatible(
        self,
        circuit: Circuit,
        placement: list[int] | None = None,
    ) -> bool:
        """Check if a circuit is compatible with this model."""
        if circuit.num_qudits > self.num_qudits:
            return False

        if any(g not in self.gate_set for g in circuit.gate_set):
            return False

        if any(e not in self.coupling_graph for e in circuit.coupling_graph):
            return False

        if any(r != self.radixes[i] for i, r in enumerate(circuit.radixes)):
            return False

        return True
