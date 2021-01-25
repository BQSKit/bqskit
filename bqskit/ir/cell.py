"""
This module implements the CircuitCell class.

A Circuit is composed of a grid of CircuitCells.
A cell groups together a gate and its parameters.
"""

class CircuitCell():
    """The CircuitCell class."""

    def __init__ ( self, gate, params = [], qudit_idx = 0 ):
        """
        CircuitCell Constructor.

        Args:
            gate (Gate): The gate in this Cell.

            params (List[float]): The parameters for the gate, if any.

            qudit_idx (int): This cell's qudit index in the gate.
        """
        pass