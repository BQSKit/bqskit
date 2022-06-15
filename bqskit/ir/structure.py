"""This module implements the CircuitStructure Class."""
from __future__ import annotations

from bqskit.ir.circuit import Circuit


class CircuitStructure:
    """Stores compressed positions of gates in a circuit as a hashable type."""

    def __init__(self, circuit: Circuit) -> None:
        """
        Construct a CircuitStructure object.

        Args:
            circuit (Circuit): The circuit to store the structure of.
        """

        if not isinstance(circuit, Circuit):
            raise TypeError(f'Expected a circuit, got {type(circuit)}.')

        structure = []

        # Compress circuit without modifying input circuit
        compressed_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        for op in circuit:
            compressed_circuit.append(op)

        # Go through gates and add gate names and locations to structure array
        current_cycle = ['No Gate'] * compressed_circuit.num_qudits
        current_cycle_index = 0

        for cycle, op in compressed_circuit.operations_with_cycles():
            if cycle != current_cycle_index:
                structure.append(tuple(current_cycle))
                current_cycle = ['No Gate'] * compressed_circuit.num_qudits
                current_cycle_index = cycle

            for qudit in op.location:
                current_cycle[qudit] = str(op)

        structure.append(tuple(current_cycle))
        self.structure = tuple(structure)
        self.hash = hash(self.structure)

    def __hash__(self) -> int:
        """Get the hash of the circuit structure."""
        return self.hash

    def __eq__(self, other: object) -> bool:
        """Return true if `self` has the same structure as `other`."""
        return (
            isinstance(other, CircuitStructure)
            and self.structure == other.structure
        )
