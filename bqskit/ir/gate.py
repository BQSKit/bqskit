"""
This module implements the Gate base class.

A gate is a potentially parameterized unitary operation
that can be applied to a circuit.
"""


from bqskit import qis


class Gate( qis.Unitary ):
    """Gate Base Class."""
    
    def get_num_params ( self ):
        """Returns the number of parameters for this gate."""

    def get_radix ( self ):
        """Returns the number of orthogonal states for each qudit."""
    
    def get_gate_size ( self ):
        """Returns the number of qudits this gate acts on."""
